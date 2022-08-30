import torch
import torch.nn as nn

from regression.models import SMPL

from regression.utils.geometry import perspective_projection, batch_rodrigues

from .encoders import TransformerEncoder
from .heads import SMPLHead
from .losses import Keypoint2DLoss, Keypoint3DLoss, ParameterLoss, BetaRegLoss, SmoothnessLoss


class TemporalModel(nn.Module):

    def __init__(self, cfg):
        super(TemporalModel, self).__init__()
        self.cfg = cfg
        self.encoder = TransformerEncoder()
        self.smpl_head = SMPLHead(cfg)
        self.focal_length = self.cfg.EXTRA.FOCAL_LENGTH

        # Define loss functions
        self.smpl_keypoint_2d_loss = Keypoint2DLoss(loss_type='l1')
        self.smpl_keypoint_3d_loss = Keypoint3DLoss(loss_type='l1')
        self.smpl_parameter_loss = ParameterLoss()
        self.beta_reg_loss = BetaRegLoss()
        self.smoothness_loss = SmoothnessLoss()

        smpl_params = {k.lower(): v for k,v in dict(cfg.SMPL).items()}
        self.batch_size = 13*self.cfg.TRAIN.BATCH_SIZE
        self.smpl = SMPL(self.cfg.SMPL.MODEL_PATH,
                         batch_size=self.batch_size,
                         joint_regressor_extra=self.cfg.SMPL.JOINT_REGRESSOR_EXTRA)

    def compute_loss(self, pred_smpl_params, pred_cam, src_key_padding_mask, data):

        batch_size = pred_cam.shape[0]
        batch_size_temp, seq_len, _, _ = data['keypoints_2d'].shape
        device = pred_cam.device
        dtype = pred_cam.dtype
        focal_length = self.focal_length * torch.ones(batch_size, 2, device=device, dtype=dtype)
        if batch_size != self.batch_size:
            self.smpl = SMPL(self.cfg.SMPL.MODEL_PATH,
                             batch_size=batch_size,
                             joint_regressor_extra=self.cfg.SMPL.JOINT_REGRESSOR_EXTRA).to(device)
            self.batch_size = batch_size

        # Get annotations and compute losses
        gt_keypoints_2d = data['keypoints_2d'].view(batch_size_temp*seq_len, -1, 3)
        gt_keypoints_3d = data['keypoints_3d'].view(batch_size_temp*seq_len, -1, 4)
        gt_smpl_params = {k: v.view(batch_size_temp*seq_len, -1) for k,v in data['smpl_params'].items()}
        has_smpl_params = {k: v.view(-1) for k,v in data['has_smpl_params'].items()}
        is_axis_angle = {k: v.view(-1) for k,v in data['smpl_params_is_axis_angle'].items()}

        smpl_output = self.smpl(**{k: v.float() for k,v in pred_smpl_params.items()}, pose2rot=False)

        pred_vertices = smpl_output.vertices
        pred_joints = smpl_output.joints

        pred_cam_t = torch.stack([pred_cam[:,1],
                                  pred_cam[:,2],
                                  2*focal_length[:, 0]/(self.cfg.MODEL.IMAGE_SIZE * pred_cam[:,0] +1e-9)],dim=-1)

        camera_center = torch.zeros(batch_size, 2, device=device, dtype=dtype)
        pred_keypoints_2d = perspective_projection(pred_joints,
                                                   rotation=torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1),
                                                   translation=pred_cam_t,
                                                   focal_length=focal_length / self.cfg.MODEL.IMAGE_SIZE,
                                                   camera_center=camera_center)

        # Compute loss on SMPL parameters
        loss_smpl_params = {}
        for k, pred in pred_smpl_params.items():
            gt = gt_smpl_params[k]
            if is_axis_angle[k].all():
                gt = batch_rodrigues(gt.view(-1,3)).view(batch_size, -1, 3, 3)
            has_gt = has_smpl_params[k]
            if pred is not None:
                loss_smpl_params[k] = self.smpl_parameter_loss(pred, gt, has_gt)
            else:
                loss_smpl_params[k] = torch.tensor(0., dtype=dtype, device=device)

        # Compute 2D keypoint loss
        loss_keypoints_2d = self.smpl_keypoint_2d_loss(pred_keypoints_2d, gt_keypoints_2d)

        # Compute 3D keypoint loss
        loss_keypoints_3d = self.smpl_keypoint_3d_loss(pred_joints, gt_keypoints_3d)

        # Beta regularization loss
        loss_beta_reg = self.beta_reg_loss(pred_smpl_params['betas'], 1 - has_smpl_params['betas'].to(dtype=pred_smpl_params['betas'].dtype))

        # Smoothness loss on parameters
        body_pose = pred_smpl_params['body_pose'].view(batch_size_temp, seq_len, -1)
        loss_smooth_parameters = self.smoothness_loss(body_pose, ~src_key_padding_mask)

        # Canonical model
        pred_smpl_params_can = {k: v for k,v in pred_smpl_params.items()}
        pred_smpl_params_can['global_orient'] = batch_rodrigues(torch.zeros([batch_size,3], device=device, dtype=dtype)).view(batch_size, 1, 3, 3)
        smpl_output_can = self.smpl(**{k: v.float() for k,v in pred_smpl_params_can.items()}, pose2rot=False)
        pred_joints_can = smpl_output_can.joints

        # Smoothness loss on 3D keypoints
        keypoints_3d = pred_joints_can.view(batch_size_temp, seq_len, -1, 3)
        loss_smooth_keypoints_3d = self.smoothness_loss(keypoints_3d, ~src_key_padding_mask)

        loss = self.cfg.LOSS_WEIGHTS['KEYPOINTS_2D'] * loss_keypoints_2d+\
               self.cfg.LOSS_WEIGHTS['KEYPOINTS_3D'] * loss_keypoints_3d+\
               sum([loss_smpl_params[k] * self.cfg.LOSS_WEIGHTS[k.upper()] for k in loss_smpl_params])+\
               self.cfg.LOSS_WEIGHTS['BETA_REG'] * loss_beta_reg+\
               self.cfg.LOSS_WEIGHTS['SMOOTH_KEYPOINTS_3D'] * loss_smooth_keypoints_3d+\
               self.cfg.LOSS_WEIGHTS['SMOOTH_BODY_POSE'] * loss_smooth_parameters

        losses = dict(loss=loss.detach(),
                      loss_keypoints_2d=loss_keypoints_2d.detach(),
                      loss_keypoints_3d=loss_keypoints_3d.detach(),
                      loss_beta_reg=loss_beta_reg,
                      loss_smooth_keypoints_3d=loss_smooth_keypoints_3d.detach(),
                      loss_smooth_parameters=loss_smooth_parameters.detach())
        
        # Add SMPL param losses to output losses
        for k, v in loss_smpl_params.items():
            losses['loss_' + k] = v.detach()

        output = {'pred_vertices': pred_vertices.detach(),
                  'pred_keypoints_2d': pred_keypoints_2d.detach(),
                  'pred_keypoints_3d': pred_joints.detach(),
                  'pred_cam_t': pred_cam_t.detach(),
                  'losses': losses,
                  'focal_length': focal_length.detach()}

        return loss, output

    def forward(self, x, src_key_padding_mask, compute_loss=False, data=None):
        batch_size, seqlen = x.shape[:2]
        feats = self.encoder(x, src_key_padding_mask)
        feats = feats.contiguous().view(-1, feats.size(-1))
        pred_smpl_params, pred_cam, pred_smpl_params_list = self.smpl_head(feats)
        if not compute_loss:
            return pred_smpl_params, pred_cam, pred_smpl_params_list
        else:
            loss, output = self.compute_loss(pred_smpl_params, pred_cam, src_key_padding_mask, data)
            return pred_smpl_params, pred_cam, pred_smpl_params_list, loss, output

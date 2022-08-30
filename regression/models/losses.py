import torch
import torch.nn as nn

class Keypoint2DLoss(nn.Module):

    def __init__(self, loss_type='l1'):
        super(Keypoint2DLoss, self).__init__()
        if loss_type == 'l1':
            self.loss_fn = nn.L1Loss(reduction='none')
        elif loss_type == 'l2':
            self.loss_fn = nn.MSELoss(reduction='none')
        else:
            raise NotImplementedError('Unsupported loss function')

    def forward(self, pred_keypoints_2d, gt_keypoints_2d):
        """ Compute 2D reprojection loss on the keypoints.
        The loss is weighted by the confidence.
        """
        conf = gt_keypoints_2d[:, :, -1].unsqueeze(-1).clone()
        loss = (conf * self.loss_fn(pred_keypoints_2d, gt_keypoints_2d[:, :, :-1])).sum() / conf.shape[0]
        return loss

class Keypoint3DLoss(nn.Module):

    def __init__(self, loss_type='l1', pelvis_ind=8):
        super(Keypoint3DLoss, self).__init__()
        if loss_type == 'l1':
            self.loss_fn = nn.L1Loss(reduction='none')
        elif loss_type == 'l2':
            self.loss_fn = nn.MSELoss(reduction='none')
        else:
            raise NotImplementedError('Unsupported loss function')

    def forward(self, pred_keypoints_3d, gt_keypoints_3d):
        """ Compute 2D reprojection loss on the keypoints.
        The loss is weighted by the confidence.
        """
        gt_keypoints_3d = gt_keypoints_3d.clone()
        pred_keypoints_3d = pred_keypoints_3d - pred_keypoints_3d[:, -5, :].unsqueeze(dim=1)
        gt_keypoints_3d[:, :, :-1] = gt_keypoints_3d[:, :, :-1] - gt_keypoints_3d[:, -5, :-1].unsqueeze(dim=1)
        conf = gt_keypoints_3d[:, :, -1].unsqueeze(-1).clone()
        gt_keypoints_3d = gt_keypoints_3d[:, :, :-1]
        loss = (conf * self.loss_fn(pred_keypoints_3d, gt_keypoints_3d)).sum() / conf.shape[0]
        return loss

class ParameterLoss(nn.Module):

    def __init__(self):
        super(ParameterLoss, self).__init__()
        self.loss_fn = nn.MSELoss(reduction='none')

    def forward(self, pred_param, gt_param, has_param):
        batch_size = pred_param.shape[0]
        num_dims = len(pred_param.shape)
        mask_dimension = [batch_size] + [1] * (num_dims-1)
        has_param = has_param.type(pred_param.type()).view(*mask_dimension)
        batch_size = has_param.shape[0]
        loss_param = (has_param * self.loss_fn(pred_param, gt_param)).sum() / batch_size
        return loss_param

class BetaRegLoss(nn.Module):

    def __init__(self):
        super(BetaRegLoss, self).__init__()
        self.loss_fn = nn.MSELoss(reduction='none')

    def forward(self, pred_betas, mask):
        return ((mask.view(-1, 1)) * (pred_betas ** 2)).sum() / pred_betas.shape[0]

class SmoothnessLoss(nn.Module):

    def __init__(self):
        super(SmoothnessLoss, self).__init__()
        self.loss_fn = nn.MSELoss(reduction='none')

    def forward(self, x, mask):
        batch_size_temp, seq_len = x.shape[:2]
        batch_size = batch_size_temp * (seq_len - 1)
        x_1 = x[:,:-1].contiguous()
        x_2 = x[:,1:].contiguous()
        mask = mask.unsqueeze(-1).unsqueeze(-1)
        mask = mask[:,:-1] * mask[:,1:]
        if len(x.shape) == 3:
            x_1 = x_1.view(batch_size, -1)
            x_2 = x_2.view(batch_size, -1)
            mask = mask.view(batch_size, -1)
        elif len(x.shape) == 4:
            x_1 = x_1.view(batch_size, -1, 3)
            x_2 = x_2.view(batch_size, -1, 3)
            mask = mask.view(batch_size, 1, 1)
        loss = (mask * self.loss_fn(x_1, x_2)).sum() / batch_size
        return loss

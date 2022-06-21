import torch
from pytorch_io import Project, BaseTrainer

from regression.models import create_model
from regression.datasets import Dataset
from regression.utils import Renderer

class SMPLTrainer(BaseTrainer):

    def init_fn(self):
        # initialize model and optimizer
        model = create_model(self.cfg)
        self.model = model.to(self.device)
        self.optimizer = torch.optim.AdamW(params=self.model.parameters(),
                                          lr=self.cfg.TRAIN.LR,
                                          weight_decay=self.cfg.TRAIN.WEIGHT_DECAY)
        self.train_ds = Dataset(self.cfg, train=True)
        self.models_dict = {'model': self.model}
        self.optimizers_dict = {'optimizer': self.optimizer}

        # Setup renderer
        if self.should_log:
            self.renderer = Renderer(self.cfg, faces=self.model.smpl.faces)

    def train_step(self, batch):
        # single training step
        self.model.train()
        images = batch['img']
        pred_smpl_params, pred_cam_t, pred_smpl_params_list, loss, output = self.model(images, compute_loss=True, data=batch) 
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()        
        output['losses']['loss'] = loss.detach()
        return output

    def train_summaries(self, batch, out, mode='train'):
        # log losses and images
        step_count = self._train_state['step_count']
        losses = out['losses']
        for loss_name, val in losses.items():
            if self.should_log:
                self.summary_writer.add_scalar(mode + '/' + loss_name, val.item(), step_count)

        # De-normalize images
        images = batch['img']
        images = images * torch.tensor([0.229, 0.224, 0.225], device=images.device).reshape(1,3,1,1)
        images = images + torch.tensor([0.485, 0.456, 0.406], device=images.device).reshape(1,3,1,1)
        images = images.cpu().numpy()
    
        pred_vertices = out['pred_vertices'].cpu().numpy()
        pred_keypoints_2d = out['pred_keypoints_2d'].cpu().numpy()
        gt_keypoints_2d = batch['keypoints_2d'].cpu().numpy()
        focal_length = out['focal_length'].cpu().numpy()
        pred_cam_t = out['pred_cam_t'].cpu().numpy()
        batch_size = images.shape[0]
        n = min(batch_size, 4)

        predictions = self.renderer.visualize_tensorboard(pred_vertices[:n], pred_cam_t[:n], images[:n], pred_keypoints_2d[:n], gt_keypoints_2d[:n], focal_length=focal_length[:n])
        if self.should_log:
            self.summary_writer.add_image('%s/predictions' % mode, predictions, step_count)

class SMPL(Project):

    def __init__(self):
        self.trainer = None

    def get_runnable(self, config):

        self.trainer = SMPLTrainer(config)
        return self.trainer.train

import torch
from pytorch_io import Project, BaseTrainer

from regression.models import create_temporal_model
from regression.datasets import TemporalDataset
from regression.utils import Renderer

class SMPLTempTrainer(BaseTrainer):

    def init_fn(self):
        # initialize model and optimizer
        model = create_temporal_model(self.cfg)
        self.model = model.to(self.device)
        self.optimizer = torch.optim.AdamW(params=self.model.parameters(),
                                          lr=self.cfg.TRAIN.LR,
                                          weight_decay=self.cfg.TRAIN.WEIGHT_DECAY)
        self.train_ds = TemporalDataset(self.cfg, train=True)
        self.models_dict = {'model': self.model}
        self.optimizers_dict = {'optimizer': self.optimizer}
        
        # Setup renderer
        if self.should_log:
            self.renderer = Renderer(self.cfg, faces=self.model.smpl.faces)

    def train_step(self, batch):
        # single training step
        self.model.train()
        feats = batch['feats']
        src_key_padding_mask = batch['src_key_padding_mask']
        pred_smpl_params, pred_cam_t, pred_smpl_params_list, loss, output = self.model(feats, src_key_padding_mask, compute_loss=True, data=batch)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        output['losses']['loss'] = loss.detach()
        return output

    def train_summaries(self, batch, out, mode='train'):
        # log losses
        step_count = self._train_state['step_count']
        losses = out['losses']
        for loss_name, val in losses.items():
            if self.should_log:
                self.summary_writer.add_scalar(mode + '/' + loss_name, val.item(), step_count)

class SMPLTemp(Project):

    def __init__(self):
        self.trainer = None

    def get_runnable(self, config):

        self.trainer = SMPLTempTrainer(config)
        return self.trainer.train

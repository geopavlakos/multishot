from .pose_dataset import PoseDataset
from .temporal_pose_dataset import TemporalPoseDataset
from .demo_dataset import DemoDataset

from typing import Optional, Union

import torch
import os
import numpy as np

class Dataset:

    def __init__(self, cfg, train=True):
        if train:
            self.datasets = [PoseDataset(cfg, **{kk.lower(): vv for kk, vv in v.items()}, img_size=cfg.MODEL.IMAGE_SIZE, focal_length=cfg.EXTRA.FOCAL_LENGTH) for k,v in cfg.DATASETS.TRAIN.items()]
            self.weights = np.array([v['WEIGHT'] for k,v in cfg.DATASETS.TRAIN.items()]).cumsum()

    def __len__(self):
        return sum([len(d) for d in self.datasets])

    def __getitem__(self, i):
        p = torch.rand(1).item()
        for i in range(len(self.datasets)):
            if p <= self.weights[i]:
                p = torch.randint(0, len(self.datasets[i]), (1,)).item()
                return self.datasets[i][p]

class TemporalDataset:

    def __init__(self, cfg, train=True):
        if train:
            self.datasets = [TemporalPoseDataset(cfg, **{kk.lower(): vv for kk, vv in v.items()}, img_size=cfg.MODEL.IMAGE_SIZE, focal_length=cfg.EXTRA.FOCAL_LENGTH) for k,v in cfg.DATASETS.TRAIN.items()]
            self.weights = np.array([v['WEIGHT'] for k,v in cfg.DATASETS.TRAIN.items()]).cumsum()

    def __len__(self):
        return sum([len(d) for d in self.datasets])

    def __getitem__(self, i):
        p = torch.rand(1).item()
        for i in range(len(self.datasets)):
            if p <= self.weights[i]:
                p = torch.randint(0, len(self.datasets[i]), (1,)).item()
                return self.datasets[i][p]

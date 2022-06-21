import torch
import pickle
import numpy as np
import smplx
from smplx import SMPL as _SMPL
from smplx.body_models import ModelOutput
from smplx.lbs import vertices2joints

class SMPL(_SMPL):
    """ Extension of the official SMPL implementation to support more joints """

    def __init__(self, *args, joint_regressor_extra, **kwargs):
        super(SMPL, self).__init__(*args, **kwargs)
        smpl_to_openpose = [24, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4,
                            7, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]
        if joint_regressor_extra is not None:
            self.register_buffer('joint_regressor_extra', torch.tensor(pickle.load(open(joint_regressor_extra, 'rb'), encoding='latin1'), dtype=torch.float32))
        self.register_buffer('joint_map', torch.tensor(smpl_to_openpose, dtype=torch.long))

    def forward(self, *args, **kwargs):
        kwargs['get_skin'] = True
        smpl_output = super(SMPL, self).forward(*args, **kwargs)
        joints = smpl_output.joints[:, self.joint_map, :]
        extra_joints = vertices2joints(self.joint_regressor_extra, smpl_output.vertices)
        joints = torch.cat([joints, extra_joints], dim=1)
        output = ModelOutput(vertices=smpl_output.vertices,
                             global_orient=smpl_output.global_orient,
                             body_pose=smpl_output.body_pose,
                             joints=joints,
                             betas=smpl_output.betas,
                             full_pose=smpl_output.full_pose)
        return output

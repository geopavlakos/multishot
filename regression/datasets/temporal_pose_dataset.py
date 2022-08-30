import torch
import os
from os.path import join
import numpy as np
import copy

from .utils import get_example, get_example_noimg


class TemporalPoseDataset:

    def __init__(self, cfg, dataset_file=None, img_dir=None, img_size=256, train=True, focal_length=None, ignore_labels=False, **kwargs):
        super(TemporalPoseDataset, self).__init__()
        self.train = train
        self.cfg = cfg
        self.ignore_labels = ignore_labels
        self.excrop = cfg.TRAIN.EXTREME_CROPPING

        self.img_size = img_size
        self.mean = np.array([123.675, 116.280, 103.530])
        self.std = np.array([58.395, 57.120, 57.375])

        self.img_dir = img_dir
        self.data = np.load(dataset_file)

        self.imgname = self.data['imgname']

        self.model_type = cfg.SMPL.MODEL_TYPE
        self.focal_length = focal_length

        body_permutation = [0, 1, 5, 6, 7, 2, 3, 4, 8, 12, 13, 14, 9, 10, 11, 16, 15, 18, 17, 22, 23, 24, 19, 20, 21]
        extra_permutation = [5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6, 12, 13, 14, 15, 16, 17, 18]
        extra_permutation = [len(body_permutation) + i for i in extra_permutation]
        flip_keypoint_permutation = body_permutation + extra_permutation
        self.flip_keypoint_permutation = flip_keypoint_permutation

        num_pose = 3 * (self.cfg.SMPL.NUM_BODY_JOINTS + 1)

        # Bounding boxes are assumed to be in the center and scale format
        self.center = self.data['center']
        self.scale = self.data['scale'].reshape(len(self.center), -1).max(axis=-1) / 200
        
        # temporal features
        self.feats = self.data['feats']
        # sequences
        self.seqs = self.data['seqs']

        # Get gt SMPL parameters, if available
        try:
            self.body_pose = self.data['body_pose'].astype(np.float)
            self.has_body_pose = self.data['has_body_pose'].astype(np.float)
        except KeyError:
            self.body_pose = np.zeros((len(self.imgname), num_pose))
            self.has_body_pose = np.zeros(len(self.imgname))
        try:
            self.betas = self.data['betas'].astype(np.float)
            self.has_betas = self.data['has_betas'].astype(np.float)
        except KeyError:
            self.betas = np.zeros((len(self.imgname), 10))
            self.has_betas = np.zeros(len(self.imgname))

        # Try to get 3D keypoints, if available
        try:
            self.body_keypoints_3d = self.data['body_keypoints_3d']
        except KeyError:
            self.body_keypoints_3d = np.zeros((len(self.imgname), 25, 4))

        # Remove problematic keypoints

        # Try to get extra 3D keypoints, if available
        try:
            self.extra_keypoints_3d = self.data['extra_keypoints_3d']
        except KeyError:
            self.extra_keypoints_3d = np.zeros((len(self.imgname), 19, 4))

        self.keypoints_3d = np.concatenate((self.body_keypoints_3d, self.extra_keypoints_3d), axis=1)

        # Try to get 2D keypoints, if available
        try:
            self.body_keypoints_2d = self.data['body_keypoints_2d']
        except KeyError:
            self.body_keypoints_2d = np.zeros((len(self.imgname), 25, 3))

        # Try to get extra 2D keypoints, if available
        try:
            self.extra_keypoints_2d = self.data['extra_keypoints_2d']
        except KeyError:
            self.extra_keypoints_2d = np.zeros((len(self.imgname), 19, 3))

        self.keypoints_2d = np.concatenate((self.body_keypoints_2d, self.extra_keypoints_2d), axis=1)


    def __len__(self):
        return self.seqs.shape[0]


    def __getitem__(self, idx):
        seq_ids = self.seqs[idx]
        img_ = []
        feats_ = []
        attend_ = []
        keypoints_2d_ = []
        keypoints_3d_ = []
        has_labels_ = []
        smpl_params_ = {'global_orient': [],
                        'body_pose': [],
                        'betas': [],
                        }
        has_smpl_params_ = {'global_orient': [],
                            'body_pose': [],
                            'betas': [],
                            }
        smpl_params_is_axis_angle_ = {'global_orient': [],
                                      'body_pose': [],
                                      'betas': [],
                                      }

        for i, idx_ in enumerate(seq_ids):
            item_ = self.get_single_frame(idx_)
            img_.append(item_['img'])
            feats_.append(item_['feats'])
            attend_.append(item_['attend'])
            keypoints_2d_.append(item_['keypoints_2d'])
            keypoints_3d_.append(item_['keypoints_3d'])
            has_labels_.append(item_['has_labels'])
            for k,v in item_['smpl_params'].items():
                smpl_params_[k].append(v)
            for k,v in item_['has_smpl_params'].items():
                has_smpl_params_[k].append(v)
            for k,v in item_['smpl_params_is_axis_angle'].items():
                smpl_params_is_axis_angle_[k].append(v)
        item = {}
        item['img'] = np.stack(img_)
        item['idx'] = idx
        item['feats'] = np.stack(feats_)
        item['src_key_padding_mask'] = np.stack(attend_)
        item['keypoints_2d'] = np.stack(keypoints_2d_)
        item['keypoints_3d'] = np.stack(keypoints_3d_)
        item['has_labels'] = np.stack(has_labels_)
        item['smpl_params'] = {k: np.stack(v) for k,v in smpl_params_.items()}
        item['has_smpl_params'] = {k: np.stack(v) for k,v in has_smpl_params_.items()}
        item['smpl_params_is_axis_angle'] = {k: np.stack(v) for k,v in smpl_params_is_axis_angle_.items()}
        return item


    def get_single_frame(self, idx):
        idx = int(idx)
        idx_ = idx
        if idx_ == -1:
            idx = 1
        try:
            image_file = join(self.img_dir, self.imgname[idx].decode('utf-8'))
        except AttributeError:
            image_file = join(self.img_dir, self.imgname[idx])
        keypoints_2d = self.keypoints_2d[idx].copy()
        keypoints_3d = self.keypoints_3d[idx].copy()
        feats = self.feats[idx].copy().astype(np.float32)

        center = self.center[idx].copy()
        center_x = center[0]
        center_y = center[1]
        bbox_size = self.scale[idx]*200
        body_pose = self.body_pose[idx].copy().astype(np.float32)
        betas = self.betas[idx].copy().astype(np.float32)

        has_body_pose = self.has_body_pose[idx].copy()
        has_betas = self.has_betas[idx].copy()

        if idx_ == -1:
            has_betas = 0
            has_body_pose = 0
            feats[:] = 0.
            attend = np.array(1).astype(np.bool)
            image_file = ''
        else:
            attend = np.array(0).astype(np.bool)

        smpl_params = {'global_orient': body_pose[:3],
                       'body_pose': body_pose[3:],
                       'betas': betas,
                       }
        has_smpl_params = {'global_orient': has_body_pose,
                           'body_pose': has_body_pose,
                           'betas': has_betas,
                           }
        smpl_params_is_axis_angle = {'global_orient': True,
                                     'body_pose': True,
                                     'betas': False,
                                     }
        keypoints_2d[:, -1] = keypoints_2d[:, -1] > 0
        keypoints_3d[:, -1] = keypoints_3d[:, -1] > 0

        keypoints_2d, keypoints_3d, smpl_params, has_smpl_params = get_example_noimg(center_x, center_y,
                                                                                     bbox_size, bbox_size,
                                                                                     keypoints_2d, keypoints_3d,
                                                                                     smpl_params, has_smpl_params,
                                                                                     self.flip_keypoint_permutation,
                                                                                     self.img_size, self.img_size,
                                                                                     self.mean, self.std, False, False)
        img_patch = np.zeros([3,1,1])

        if smpl_params['body_pose'].shape[0] != 69:
            import ipdb; ipdb.set_trace()
        if self.model_type == 'smpl':
            keypoints_2d = np.concatenate((keypoints_2d[:25], keypoints_2d[-19:]), axis=0)
            keypoints_3d = np.concatenate((keypoints_3d[:25], keypoints_3d[-19:]), axis=0)
        keypoints_3d[[1, 8, 9, 12], -1] = 0.
        keypoints_2d[[1, 8, 9, 12], -1] = 0.
        keypoints_3d[[-19, -18, -15, -14, -13, -12, -11, -10, -9, -8], -1] = 0.
        keypoints_2d[[-19, -18, -15, -14, -13, -12, -11, -10, -9, -8], -1] = 0.
        item = {}
        item['img'] = img_patch
        item['feats'] = feats
        item['attend'] = attend
        item['keypoints_2d'] = keypoints_2d.astype(np.float32)
        item['keypoints_3d'] = keypoints_3d.astype(np.float32)
        item['smpl_params'] = smpl_params
        item['has_smpl_params'] = has_smpl_params
        item['smpl_params_is_axis_angle'] = smpl_params_is_axis_angle
        if self.ignore_labels:
            item['has_smpl_params'] = {k: 0*v for k,v in item['has_smpl_params'].items()}
            item['keypoints_2d'][:, -1] = 0.
            item['keypoints_3d'][:, -1] = 0.
        item['has_labels'] = 0 if self.ignore_labels else 1
        return item

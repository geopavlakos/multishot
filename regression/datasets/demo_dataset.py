import os
import torch
import numpy as np
import copy

from .utils import get_example

class DemoDataset:

    def __init__(self, cfg, dataset_file=None, img_size=256, train=False, focal_length=None, ignore_labels=False, **kwargs):
        super(DemoDataset, self).__init__()
        self.train = train
        self.cfg = cfg
        self.ignore_labels = ignore_labels

        self.img_size = img_size
        self.mean = np.array([123.675, 116.280, 103.530])
        self.std = np.array([58.395, 57.120, 57.375])

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
        self.scale = self.data['scale']*200
        
        # prepare sequences
        valid = self.data['valid']
        self.seqs = -np.ones([np.sum(valid), 13])
        counter = 0
        for frame_i, valid_i in enumerate(valid):
            if valid_i == 0:
                continue
            seqs_ = -np.ones(13)
            for j, frame_j in enumerate(range(frame_i-6, frame_i+7)):
                if frame_j < 0 or frame_j >= len(valid):
                    continue
                if valid[frame_j] == 0:
                    continue
                seqs_[j] = frame_j
            self.seqs[counter] = seqs_
            counter = counter + 1

        # Try to get 2D keypoints, if available
        try:
            self.body_keypoints_2d = self.data['body_keypoints']
        except KeyError:
            self.body_keypoints_2d = np.zeros((len(self.imgname), 25, 3))
        self.extra_keypoints_2d = np.zeros((len(self.imgname), 19, 3))
        self.keypoints_2d = np.concatenate((self.body_keypoints_2d, self.extra_keypoints_2d), axis=1)


    def __len__(self):
        return self.seqs.shape[0]


    def __getitem__(self, idx):
        seq_ids = self.seqs[idx]
        img_ = []
        attend_ = []
        for i, idx_ in enumerate(seq_ids):
            item_ = self.get_single_frame(idx_)
            img_.append(item_['img'])
            attend_.append(item_['attend'])
        item = {}
        item['img'] = np.stack(img_)
        item['idx'] = idx
        item['src_key_padding_mask'] = np.stack(attend_)
        return item


    def get_single_frame(self, idx):
        idx = int(idx)
        idx_ = idx
        if idx_ == -1:
            idx = 1
        try:
            image_file = self.imgname[idx].decode('utf-8')
        except AttributeError:
            image_file = self.imgname[idx]
        keypoints_2d = self.keypoints_2d[idx].copy()
        keypoints_3d = np.zeros([44, 4])

        center = self.center[idx].copy()
        center_x = center[0]
        center_y = center[1]
        bbox_size = self.scale[idx]

        if idx_ == -1:
            attend = np.array(1).astype(np.bool)
            image_file = ''
        else:
            attend = np.array(0).astype(np.bool)

        smpl_params = {'global_orient': np.zeros(3),
                       'body_pose': np.zeros(69),
                       'betas': np.zeros(10),
                       }
        has_smpl_params = {'global_orient': False,
                           'body_pose': False,
                           'betas': False,
                           }
        smpl_params_is_axis_angle = {'global_orient': True,
                                     'body_pose': True,
                                     'betas': False,
                                     }
        keypoints_2d[:, -1] = keypoints_2d[:, -1] > 0
        keypoints_3d[:, -1] = keypoints_3d[:, -1] > 0

        img_patch, _, _, _, _ = get_example(image_file,
                                            center_x, center_y,
                                            bbox_size, bbox_size,
                                            keypoints_2d, keypoints_3d,
                                            smpl_params, has_smpl_params,
                                            self.flip_keypoint_permutation,
                                            self.img_size, self.img_size,
                                            self.mean, self.std, False, False)

        item = {}
        item['img'] = img_patch
        item['attend'] = attend
        return item

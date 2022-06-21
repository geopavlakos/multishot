# File adapted from: https://github.com/vchoutas/smplify-x

# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems and the Max Planck Institute for Biological
# Cybernetics. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import os.path as osp

import time
import yaml
import torch
from tqdm import tqdm

import numpy as np
import glob
import pickle
import cv2
import PIL.Image as pil_img

import trimesh
import pyrender

import smplx

from utils import JointMapper
from cmd_parser import parse_config
from data_parser import create_dataset
from fit_single_frame import fit_single_frame

from camera import create_camera
from prior import create_prior

torch.backends.cudnn.enabled = False

def render_fittings(**args):
    output_folder = args.pop('output_folder')

    print(output_folder)

    float_dtype = args['float_dtype']
    if float_dtype == 'float64':
        dtype = torch.float64
    elif float_dtype == 'float32':
        dtype = torch.float64
    else:
        print('Unknown float type {}, exiting!'.format(float_dtype))
        sys.exit(-1)

    use_cuda = args.get('use_cuda', True)
    if use_cuda and not torch.cuda.is_available():
        print('CUDA is not available, exiting!')
        sys.exit(-1)

    img_folder = args.pop('img_folder', 'images')

    start = time.time()

    input_gender = args.pop('gender', 'neutral')
    gender_lbl_type = args.pop('gender_lbl_type', 'none')

    float_dtype = args.get('float_dtype', 'float32')
    if float_dtype == 'float64':
        dtype = torch.float64
    elif float_dtype == 'float32':
        dtype = torch.float32
    else:
        raise ValueError('Unknown float type {}, exiting!'.format(float_dtype))

    model_params = dict(model_path=args.get('model_folder'),
                        #joint_mapper=joint_mapper,
                        create_global_orient=True,
                        create_body_pose=not args.get('use_vposer'),
                        create_betas=True,
                        create_left_hand_pose=True,
                        create_right_hand_pose=True,
                        create_expression=True,
                        create_jaw_pose=True,
                        create_leye_pose=True,
                        create_reye_pose=True,
                        create_transl=False,
                        dtype=dtype,
                        **args)

    model_type = args.get('model_type')
    neutral_model = smplx.create(gender='neutral', **model_params)

    use_hands = args.get('use_hands', True)
    use_face = args.get('use_face', True)
    focal_length = args.get('focal_length')

    use_cuda = False
    if use_cuda and torch.cuda.is_available():
        device = torch.device('cuda')

        angle_prior = angle_prior.to(device=device)
        shape_prior = shape_prior.to(device=device)
        if use_face:
            expr_prior = expr_prior.to(device=device)
            jaw_prior = jaw_prior.to(device=device)
        if use_hands:
            left_hand_prior = left_hand_prior.to(device=device)
            right_hand_prior = right_hand_prior.to(device=device)
    else:
        device = torch.device('cpu')
    
    body_model = neutral_model

    npz_contents = np.load(args.get('npz'))
    H_ = npz_contents['height']
    W_ = npz_contents['width']
    imgname_ = npz_contents['imgname']
    valid_ = npz_contents['valid']
    pkls = glob.glob(os.path.join(output_folder, 'results', '*.pkl'))
    pkls.sort()
    
    counter = -1
    for pkl_i in pkls:
        pkl_contents = pickle.load(open(pkl_i, 'rb'))
        len_pkl = pkl_contents['body_pose'].shape[0]
        body_pose_ = pkl_contents['body_pose']
        camera_translation_ = pkl_contents['camera_translation']
        global_orient_ = pkl_contents['global_orient']
        betas_ = pkl_contents['betas']
        camera_rotation_ = pkl_contents['camera_rotation']
        for j in range(len_pkl):
            counter += 1
            print('Rendering image %06d of %06d' % (counter+1, len(H_)))
            if valid_[counter] == 0:
                continue
            body_pose = body_pose_[j]
            camera_transl = camera_translation_[j]
            global_orient = global_orient_[j]
            betas = betas_[j]

            body_model.reset_params(body_pose=body_pose,
                                    betas=betas,
                                    global_orient=global_orient)
            
            W = W_[counter]
            H = H_[counter]
            imgname = imgname_[counter]
            img = cv2.imread(imgname).astype(np.float32)[:, :, ::-1] / 255.0

            camera_center = np.array([W/2., H/2.])
            batch_size, curr_batch_size = 1, 1

            curr_mesh_folder = osp.join(output_folder, 'meshes')
            if not osp.exists(curr_mesh_folder):
                os.makedirs(curr_mesh_folder)
            curr_img_folder = osp.join(output_folder, 'images')
            if not osp.exists(curr_img_folder):
                os.makedirs(curr_img_folder)

            out_img_fn = osp.join(curr_img_folder, 'frame_%05d.jpg' % counter)
            mesh_fn = osp.join(curr_mesh_folder, '%05d.obj' % counter)

            model_output = body_model(return_verts=True)
            vertices = model_output.vertices.detach().numpy().squeeze()

            out_mesh = trimesh.Trimesh(vertices, body_model.faces, process=False)
            rot = trimesh.transformations.rotation_matrix(
                np.radians(180), [1, 0, 0])
            out_mesh.apply_transform(rot)
            out_mesh.export(mesh_fn)

            material = pyrender.MetallicRoughnessMaterial(
                metallicFactor=0.0,
                alphaMode='OPAQUE',
                baseColorFactor=(1.0, 1.0, 0.9, 1.0))
            mesh = pyrender.Mesh.from_trimesh(
                out_mesh,
                material=material)
    
            scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
                                   ambient_light=(0.3, 0.3, 0.3))
            scene.add(mesh, 'mesh')
    
            # Equivalent to 180 degrees around the y-axis. Transforms the fit to
            # OpenGL compatible coordinate system.
            camera_transl[0] *= -1.0

            camera_pose = np.eye(4)
            camera_pose[:3, 3] = camera_transl
    
            camera_ = pyrender.camera.IntrinsicsCamera(
                fx=focal_length, fy=focal_length,
                cx=camera_center[0], cy=camera_center[1])
            scene.add(camera_, pose=camera_pose)
    
            light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1)
            light_pose = np.eye(4)
    
            light_pose[:3, 3] = np.array([0, -1, 1])
            scene.add(light, pose=light_pose)
    
            light_pose[:3, 3] = np.array([0, 1, 1])
            scene.add(light, pose=light_pose)
    
            light_pose[:3, 3] = np.array([1, 1, 2])
            scene.add(light, pose=light_pose)
    
            r = pyrender.OffscreenRenderer(viewport_width=W,
                                           viewport_height=H,
                                           point_size=1.0)
            color, depth = r.render(scene, flags=pyrender.RenderFlags.RGBA)
            color = color.astype(np.float32) / 255.0
    
            valid_mask = (color[:, :, -1] > 0)[:, :, np.newaxis]
            input_img = img
            output_img = (color[:, :, :-1] * valid_mask +
                          (1 - valid_mask) * input_img)
    
            img_ = pil_img.fromarray((output_img * 255).astype(np.uint8))
            img_.save(out_img_fn)

    elapsed = time.time() - start
    time_msg = time.strftime('%H hours, %M minutes, %S seconds',
                             time.gmtime(elapsed))
    print('Processing the data took: {}'.format(time_msg))


if __name__ == "__main__":
    args = parse_config()
    render_fittings(**args)

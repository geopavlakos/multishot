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


import time
try:
    import cPickle as pickle
except ImportError:
    import pickle

import sys
import os
import os.path as osp

import numpy as np
import torch

from tqdm import tqdm
from IPython.core.debugger import set_trace

from collections import defaultdict

import cv2
import PIL.Image as pil_img

from optimizers import optim_factory

import fitting

from smplx.lbs import transform_mat, batch_rodrigues

def fit_single_frame(img, keypoints,
                     height, width,
                     seqname, valid, shot,
                     init_betas, init_pose, cam_trans,
                     body_model,
                     camera,
                     joint_weights,
                     body_pose_prior,
                     jaw_prior,
                     left_hand_prior,
                     right_hand_prior,
                     shape_prior,
                     expr_prior,
                     angle_prior,
                     motion_prior,
                     result_fn='out.pkl',
                     mesh_fn='out.obj',
                     out_img_fn='overlay.png',
                     loss_type='smplify',
                     use_cuda=True,
                     init_joints_idxs=(9, 12, 2, 5),
                     use_face=True,
                     use_hands=True,
                     data_weights=None,
                     body_pose_prior_weights=None,
                     hand_pose_prior_weights=None,
                     jaw_pose_prior_weights=None,
                     shape_weights=None,
                     expr_weights=None,
                     hand_joints_weights=None,
                     face_joints_weights=None,
                     depth_loss_weight=1e2,
                     interpenetration=True,
                     coll_loss_weights=None,
                     df_cone_height=0.5,
                     penalize_outside=True,
                     max_collisions=8,
                     point2plane=False,
                     part_segm_fn='',
                     focal_length=5000.,
                     side_view_thsh=25.,
                     rho=100,
                     vposer_latent_dim=32,
                     vposer_ckpt='',
                     use_joints_conf=False,
                     interactive=True,
                     visualize=False,
                     save_meshes=True,
                     degrees=None,
                     batch_size=1,
                     dtype=torch.float32,
                     ign_part_pairs=None,
                     left_shoulder_idx=2,
                     right_shoulder_idx=5,
                     **kwargs):

    device = torch.device('cuda') if use_cuda else torch.device('cpu')

    if degrees is None:
        degrees = [0, 90, 180, 270]

    if data_weights is None:
        data_weights = [1, ] * 1

    if body_pose_prior_weights is None:
        body_pose_prior_weights = [4.04 * 1e2, 4.04 * 1e2, 57.4, 4.78]

    msg = (
        'Number of Body pose prior weights {}'.format(
            len(body_pose_prior_weights)) +
        ' does not match the number of data term weights {}'.format(
            len(data_weights)))
    assert (len(data_weights) ==
            len(body_pose_prior_weights)), msg

    if use_hands:
        if hand_pose_prior_weights is None:
            hand_pose_prior_weights = [1e2, 5 * 1e1, 1e1, .5 * 1e1]
        msg = ('Number of Body pose prior weights does not match the' +
               ' number of hand pose prior weights')
        assert (len(hand_pose_prior_weights) ==
                len(body_pose_prior_weights)), msg
        if hand_joints_weights is None:
            hand_joints_weights = [0.0, 0.0, 0.0, 1.0]
            msg = ('Number of Body pose prior weights does not match the' +
                   ' number of hand joint distance weights')
            assert (len(hand_joints_weights) ==
                    len(body_pose_prior_weights)), msg

    if shape_weights is None:
        shape_weights = [1e2, 5 * 1e1, 1e1, .5 * 1e1]
    msg = ('Number of Body pose prior weights = {} does not match the' +
           ' number of Shape prior weights = {}')
    assert (len(shape_weights) ==
            len(body_pose_prior_weights)), msg.format(
                len(shape_weights),
                len(body_pose_prior_weights))

    if use_face:
        if jaw_pose_prior_weights is None:
            jaw_pose_prior_weights = [[x] * 3 for x in shape_weights]
        else:
            jaw_pose_prior_weights = map(lambda x: map(float, x.split(',')),
                                         jaw_pose_prior_weights)
            jaw_pose_prior_weights = [list(w) for w in jaw_pose_prior_weights]
        msg = ('Number of Body pose prior weights does not match the' +
               ' number of jaw pose prior weights')
        assert (len(jaw_pose_prior_weights) ==
                len(body_pose_prior_weights)), msg

        if expr_weights is None:
            expr_weights = [1e2, 5 * 1e1, 1e1, .5 * 1e1]
        msg = ('Number of Body pose prior weights = {} does not match the' +
               ' number of Expression prior weights = {}')
        assert (len(expr_weights) ==
                len(body_pose_prior_weights)), msg.format(
                    len(body_pose_prior_weights),
                    len(expr_weights))

        if face_joints_weights is None:
            face_joints_weights = [0.0, 0.0, 0.0, 1.0]
        msg = ('Number of Body pose prior weights does not match the' +
               ' number of face joint distance weights')
        assert (len(face_joints_weights) ==
                len(body_pose_prior_weights)), msg

    if coll_loss_weights is None:
        coll_loss_weights = [0.0] * len(body_pose_prior_weights)
    msg = ('Number of Body pose prior weights does not match the' +
           ' number of collision loss weights')
    assert (len(coll_loss_weights) ==
            len(body_pose_prior_weights)), msg

    use_vposer = kwargs.get('use_vposer', True)
    vposer, pose_embedding = [None, ] * 2
    if use_vposer:
        pose_embedding = torch.zeros([batch_size, 32],
                                     dtype=dtype, device=device,
                                     requires_grad=True)

        vposer_ckpt = osp.expandvars(vposer_ckpt)
        vposer, _ = load_vposer(vposer_ckpt, vp_model='snapshot')
        vposer = vposer.to(device=device)
        vposer.eval()

    model_type = kwargs.get('model_type')
    motion_smooth = kwargs.get('motion_smooth')

    # regression initialization 
    init_pose = torch.tensor(init_pose, dtype=dtype)
    init_betas = torch.tensor(init_betas, dtype=dtype)
    cam_trans = torch.tensor(cam_trans, dtype=dtype)

    valid = torch.tensor(np.array(valid))
    seqname = np.array(seqname)
    shot = np.array(shot)

    both_valid = torch.tensor(valid[1:]*valid[:-1], dtype=dtype, device=device)
    same_seq = torch.tensor(np.char.equal(seqname[1:], seqname[:-1]), dtype=dtype, device=device)
    same_shot = torch.tensor(np.equal(shot[1:], shot[:-1]), dtype=dtype, device=device)

    if motion_smooth == 'multi_shot':
        motion_example_weight = both_valid * same_seq

    keypoint_data = torch.tensor(keypoints, dtype=dtype)
    gt_joints = keypoint_data[:, :, :2]
    if use_joints_conf:
        joints_conf = keypoint_data[:, :, 2]#.reshape(1, -1)
    joints_conf = joints_conf*valid[:,None].float()

    # Transfer the data to the correct device
    gt_joints = gt_joints.to(device=device, dtype=dtype)
    if use_joints_conf:
        joints_conf = joints_conf.to(device=device, dtype=dtype)

    valid_frames = torch.tensor(valid, dtype=dtype)

    # Create the search tree
    search_tree = None
    pen_distance = None
    filter_faces = None
    if interpenetration:
        from mesh_intersection.bvh_search_tree import BVH
        import mesh_intersection.loss as collisions_loss
        from mesh_intersection.filter_faces import FilterFaces

        assert use_cuda, 'Interpenetration term can only be used with CUDA'
        assert torch.cuda.is_available(), \
            'No CUDA Device! Interpenetration term can only be used' + \
            ' with CUDA'

        search_tree = BVH(max_collisions=max_collisions)

        pen_distance = \
            collisions_loss.DistanceFieldPenetrationLoss(
                sigma=df_cone_height, point2plane=point2plane,
                vectorized=True, penalize_outside=penalize_outside)

        if part_segm_fn:
            # Read the part segmentation
            part_segm_fn = os.path.expandvars(part_segm_fn)
            with open(part_segm_fn, 'rb') as faces_parents_file:
                face_segm_data = pickle.load(faces_parents_file,
                                             encoding='latin1')
            faces_segm = face_segm_data['segm']
            faces_parents = face_segm_data['parents']
            # Create the module used to filter invalid collision pairs
            filter_faces = FilterFaces(
                faces_segm=faces_segm, faces_parents=faces_parents,
                ign_part_pairs=ign_part_pairs).to(device=device)

    # Weights used for the pose prior and the shape prior
    opt_weights_dict = {'data_weight': data_weights,
                        'body_pose_weight': body_pose_prior_weights,
                        'shape_weight': shape_weights}
    if use_face:
        opt_weights_dict['face_weight'] = face_joints_weights
        opt_weights_dict['expr_prior_weight'] = expr_weights
        opt_weights_dict['jaw_prior_weight'] = jaw_pose_prior_weights
    if use_hands:
        opt_weights_dict['hand_weight'] = hand_joints_weights
        opt_weights_dict['hand_prior_weight'] = hand_pose_prior_weights
    if interpenetration:
        opt_weights_dict['coll_loss_weight'] = coll_loss_weights

    keys = opt_weights_dict.keys()
    opt_weights = [dict(zip(keys, vals)) for vals in
                   zip(*(opt_weights_dict[k] for k in keys
                         if opt_weights_dict[k] is not None))]
    for weight_list in opt_weights:
        for key in weight_list:
            weight_list[key] = torch.tensor(weight_list[key],
                                            device=device,
                                            dtype=dtype)

    # The indices of the joints used for the initialization of the camera
    init_joints_idxs = torch.tensor(init_joints_idxs, device=device)

    edge_indices = kwargs.get('body_tri_idxs')

    camera_loss = fitting.create_loss('camera_init',
                                      trans_estimation=cam_trans,
                                      init_joints_idxs=init_joints_idxs,
                                      depth_loss_weight=depth_loss_weight,
                                      dtype=dtype).to(device=device)
    camera_loss.trans_estimation[:] = cam_trans

    loss = fitting.create_loss(loss_type=loss_type,
                               joint_weights=joint_weights,
                               rho=rho,
                               use_joints_conf=use_joints_conf,
                               use_face=use_face, use_hands=use_hands,
                               vposer=vposer,
                               pose_embedding=pose_embedding,
                               valid_frames=valid,
                               body_pose_prior=body_pose_prior,
                               shape_prior=shape_prior,
                               angle_prior=angle_prior,
                               expr_prior=expr_prior,
                               left_hand_prior=left_hand_prior,
                               right_hand_prior=right_hand_prior,
                               jaw_prior=jaw_prior,
                               motion_prior=motion_prior,
                               interpenetration=interpenetration,
                               pen_distance=pen_distance,
                               search_tree=search_tree,
                               tri_filtering_module=filter_faces,
                               dtype=dtype,
                               **kwargs)
    loss = loss.to(device=device)

    with fitting.FittingMonitor(
            batch_size=batch_size, visualize=visualize, **kwargs) as monitor:

        img = torch.tensor(img, dtype=dtype)

        # good enough for most frames
        H = torch.tensor(height, dtype=dtype, device=device) 
        W = torch.tensor(width, dtype=dtype, device=device)
        data_weight = 1000 / H
        # The closure passed to the optimizer
        camera_loss.reset_loss_weights({'data_weight': data_weight})

        body_model.reset_params(body_pose=init_pose[:,3:],
                                betas=init_betas,
                                global_orient=init_pose[:,:3])

        # Reset the parameters to estimate the initial translation of the
        # body model
        try_both_orient = False

        # Update the value of the translation of the camera as well as
        # the image center.
        with torch.no_grad():
            camera.translation[:] = cam_trans.view_as(camera.translation)
            camera.center[:] = torch.cat([W[:,None], H[:,None]], dim=1) * 0.5

        # Re-enable gradient calculation for the camera translation
        camera.translation.requires_grad = True

        camera_opt_params = [camera.translation, body_model.global_orient]

        camera_optimizer, camera_create_graph = optim_factory.create_optimizer(
            camera_opt_params,
            **kwargs)

        # The closure passed to the optimizer
        fit_camera = monitor.create_fitting_closure(
            camera_optimizer, body_model, camera, gt_joints,
            camera_loss, create_graph=camera_create_graph,
            use_vposer=use_vposer, vposer=vposer,
            pose_embedding=pose_embedding,
            return_full_pose=False, return_verts=False)

        # Step 1: Optimize over the torso joints the camera translation
        # Initialize the computational graph by feeding the initial translation
        # of the camera and the initial pose of the body model.
        camera_init_start = time.time()
        cam_init_loss_val = monitor.run_fitting(camera_optimizer,
                                                fit_camera,
                                                camera_opt_params, body_model,
                                                use_vposer=use_vposer,
                                                pose_embedding=pose_embedding,
                                                vposer=vposer)

        if interactive:
            if use_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()
            tqdm.write('Camera initialization done after {:.4f}'.format(
                time.time() - camera_init_start))
            tqdm.write('Camera initialization final loss {:.4f}'.format(
                cam_init_loss_val))

        #return
        if torch.isnan(body_model.global_orient).any() or torch.isnan(camera.translation).any():
            body_orient = init_pose[:,:3].cpu().numpy()
            with torch.no_grad():
                camera.translation[:] = cam_trans.view_as(camera.translation)
        else:
            body_orient = body_model.global_orient.detach().cpu().numpy()

        # If the 2D detections/positions of the shoulder joints are too
        # close the rotate the body by 180 degrees and also fit to that
        # orientation
        if try_both_orient:
            flipped_orient = cv2.Rodrigues(body_orient)[0].dot(
                cv2.Rodrigues(np.array([0., np.pi, 0]))[0])
            flipped_orient = cv2.Rodrigues(flipped_orient)[0].ravel()

            flipped_orient = torch.tensor(flipped_orient,
                                          dtype=dtype,
                                          device=device).unsqueeze(dim=0)
            orientations = [body_orient, flipped_orient]
        else:
            orientations = [body_orient]

        # store here the final error for both orientations,
        # and pick the orientation resulting in the lowest error
        results = []

        # Step 2: Optimize the full model
        final_loss_val = 0
        for or_idx, orient in enumerate(tqdm(orientations, desc='Orientation')):
            opt_start = time.time()

            # Reset the parameters to estimate the initial translation of the
            # body model
            new_params = defaultdict(global_orient=orient,
                                     betas=init_betas,
                                     body_pose=init_pose[:,3:])
            body_model.reset_params(**new_params)

            if use_vposer:
                with torch.no_grad():
                    pose_embedding.fill_(0)

            for opt_idx, curr_weights in enumerate(tqdm(opt_weights, desc='Stage')):

                body_params = list(body_model.parameters())

                final_params = list(
                    filter(lambda x: x.requires_grad, body_params))

                if use_vposer:
                    final_params.append(pose_embedding)

                final_params.append(camera.translation)

                body_optimizer, body_create_graph = optim_factory.create_optimizer(
                    final_params,
                    **kwargs)
                body_optimizer.zero_grad()

                curr_weights['data_weight'] = data_weight
                curr_weights['bending_prior_weight'] = (
                    3.17 * curr_weights['body_pose_weight'])
                if use_hands:
                    joint_weights[:, 25:67] = curr_weights['hand_weight']
                if use_face:
                    joint_weights[:, 67:] = curr_weights['face_weight']
                loss.reset_loss_weights(curr_weights)

                closure = monitor.create_fitting_closure(
                    body_optimizer, body_model,
                    camera=camera, gt_joints=gt_joints,
                    joints_conf=joints_conf,
                    model_type=model_type,
                    motion_example_weight=motion_example_weight,
                    joint_weights=joint_weights,
                    loss=loss, create_graph=body_create_graph,
                    use_vposer=use_vposer, vposer=vposer,
                    pose_embedding=pose_embedding,
                    return_verts=True, return_full_pose=True)

                if interactive:
                    if use_cuda and torch.cuda.is_available():
                        torch.cuda.synchronize()
                    stage_start = time.time()
                final_loss_val = monitor.run_fitting(
                    body_optimizer,
                    closure, final_params,
                    body_model,
                    pose_embedding=pose_embedding, vposer=vposer,
                    use_vposer=use_vposer)

                if interactive:
                    if use_cuda and torch.cuda.is_available():
                        torch.cuda.synchronize()
                    elapsed = time.time() - stage_start
                    if interactive:
                        tqdm.write('Stage {:03d} done after {:.4f} seconds'.format(
                            opt_idx, elapsed))

            if interactive:
                if use_cuda and torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed = time.time() - opt_start
                tqdm.write(
                    'Body fitting Orientation {} done after {:.4f} seconds'.format(
                        or_idx, elapsed))
                if not(final_loss_val is None):
                    tqdm.write('Body final loss val = {:.5f}'.format(
                        final_loss_val))

            # Get the result of the fitting process
            # Store in it the errors list in order to compare multiple
            # orientations, if they exist
            result = {'camera_' + str(key): val.detach().cpu().numpy()
                      for key, val in camera.named_parameters()}
            result.update({key: val.detach().cpu().numpy()
                           for key, val in body_model.named_parameters()})

            model_output = body_model(return_verts=False)
            joints_3d = model_output.joints
            projected_joints = camera(joints_3d)
            result.update({'projected_joints': projected_joints.detach().cpu().numpy()})
            if use_vposer:
                result['body_pose'] = pose_embedding.detach().cpu().numpy()

            results.append({'loss': final_loss_val,
                            'result': result})

        with open(result_fn, 'wb') as result_file:
            if len(results) > 1:
                min_idx = (0 if results[0]['loss'] < results[1]['loss']
                           else 1)
            else:
                min_idx = 0
            pickle.dump(results[min_idx]['result'], result_file, protocol=2)

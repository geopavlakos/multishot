import os
import cv2
import json
import glob
import joblib
import argparse
import numpy as np
from scipy.optimize import linear_sum_assignment

# Defaults from PHALP
FOCAL_LENGTH = 5000
IMG_RES = 256

# Define command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--phalp_output', default='PHALP/out/folder', help='folder where the PHALP output is stored')
parser.add_argument('--phalp_demo', default='PHALP/demo/folder', help='folder that the PHALP demo creates')
parser.add_argument('--openpose_output', default='opt_output/demo', help='folder where the OpenPose output is stored')
parser.add_argument('--output_npz', default='opt_output/demo', help='folder for the multi-shot optimization')
parser.add_argument('--tracklet_id', default=1, type=int, help='which PHALP tracklet to return')

# compute matching score for mask vs keypoints
def get_mask_keypoint_cost(mask, keypoints):

    xs = keypoints[:,0]
    ys = keypoints[:,1]
    confs = keypoints[:,2]
    width = mask.shape[1]
    height = mask.shape[0]

    confs[xs>=width] = 0
    confs[ys>=width] = 0
    xs[xs>=width] = width-1
    ys[ys>=height] = height-1
    mask_pixels = mask[np.floor(ys).astype(int), np.floor(xs).astype(int)].clip(0,1)

    p_ = confs[mask_pixels>0].sum()/confs.sum() + 1e-1
    r_ = len(confs[mask_pixels>0])/25 + 1e-1
    cost = 25 - 1/(1/p_ + 1/r_)*25

    return cost

if __name__ == '__main__':
    args = parser.parse_args()

    # read PHALP results file
    pkl_file = glob.glob(os.path.join(args.phalp_output, 'results', '*.pkl'))
    tracking_results = joblib.load(pkl_file[0])
    frames = tracking_results.keys()
    
    imgnames_, widths_, heights_ = [], [], []
    scales_, centers_ = [], []
    body_keypoints_, face_keypoints_ = [], []
    pred_pose_, pred_betas_, cam_trans_, init_torso_ = [], [], [], []
    seqnames_, valid_, shots_ = [], [], []

    seqname = 'demo'
    for frame_i in frames:
        track_frame = tracking_results[frame_i]
        shot = 0

        # tracklet id is detected in this frame
        if args.tracklet_id in track_frame['tracked_ids']:
            index = track_frame['tracked_ids'].index(args.tracklet_id)
            smpl = track_frame['smpl'][index]
            global_orient = smpl['global_orient']
            body_pose = smpl['body_pose']
            betas = smpl['betas'][0]
    
            pred_cam = track_frame['camera'][index]
            joints_3d = track_frame['3d_joints'][index]
            H, W = track_frame['size'][index]
    
            imgname = os.path.join(args.phalp_demo, 'img', track_frame['img_name'][index])
    
            center = track_frame['center'][index]
            bbox = track_frame['bbox'][index]
            scale = np.max(bbox[-2:]/200)

            # bbox camera translation for projection
            pred_cam_t = np.stack([pred_cam[:,1],
                                   pred_cam[:,2],
                                   2*FOCAL_LENGTH/(IMG_RES * pred_cam[:,0] +1e-9)], axis=-1)
    
            # intrinsics
            K = np.eye(3)
            K[0,0] = FOCAL_LENGTH
            K[1,1] = FOCAL_LENGTH
    
            # projection
            points = joints_3d + pred_cam_t
            projected_points = points / points[:,[-1]]
            projected_points = np.einsum('ij,kj->ki', K, projected_points)[:, :-1]
    
            # convert keypoint locationus
            projected_points = projected_points/IMG_RES * 200 * scale + center[None]
            torso_joints = projected_points[[9,12,2,5]]
    
            # full image camera translation for smplify
            depth = 2*FOCAL_LENGTH/(200 * scale * pred_cam[0,0] + 1e-9)
            tx = pred_cam[0,1] + depth/FOCAL_LENGTH*(center[0]-W/2)
            ty = pred_cam[0,0] + depth/FOCAL_LENGTH*(center[1]-H/2)
            cam_trans = np.stack([tx, ty, depth], axis=-1)
     
            # convert pose parameters to axis-angle
            pred_pose_aa = np.zeros(72)
            pred_pose_aa[:3] = cv2.Rodrigues(global_orient[0,0])[0].T
            for j in range(23):
                pred_pose_aa[(j+1)*3:(j+2)*3] = cv2.Rodrigues(body_pose[0,j])[0].T
    
            # figure out assignment with OpenPose
            openpose_file = os.path.join(args.openpose_output, '%s_keypoints.json' % frame_i[:-4])
            json_contents = json.load(open(openpose_file, 'rb'))

            pred_masks = glob.glob(os.path.join(args.phalp_output, '_TMP', '*%s_*' % frame_i[:-4]))
            pred_masks.sort()

            # computes scores for Hungarian matching
            cost_key_mask = np.zeros((len(pred_masks), len(json_contents['people'])), dtype=np.float32)
            for i, pred_mask in enumerate(pred_masks):
                mask = cv2.imread(pred_mask, 2)
                for j, person in enumerate(json_contents['people']):
                    keypoints = np.reshape(person['pose_keypoints_2d'], [-1,3])
                    cost_key_mask[i, j] = get_mask_keypoint_cost(mask, keypoints)
            cost_key_mask[cost_key_mask>17] = 17 + 0.1
            row_ind, col_ind = linear_sum_assignment(cost_key_mask)

            for i in range(len(pred_masks)):
                if track_frame['mask_name'][index] in pred_masks[i]:
                    mask_id = i
                    break
            if mask_id in row_ind:
                # assign OpenPose keypoints to the mask of this frame
                op_id = col_ind[np.where(row_ind==mask_id)[0][0]]
                person = json_contents['people'][op_id]
                body_keypoints = np.reshape(person['pose_keypoints_2d'], [-1,3])
                valid = 1
            else:
                # could not find a match for this mask
                body_keypoints = np.zeros([25,3])
                valid = 0
        
        # tracklet id is not detected in this frame
        else:
            valid = 0
            scale = 1
            center = [0, 0]
            imgname = os.path.join(args.phalp_demo, 'img', track_frame['img_name'][0])
            H, W = track_frame['size'][0]['size']
            pred_pose_aa = np.zeros(72)
            betas = np.zeros(10)
            cam_trans = np.zeros(3)
            torso_joints = np.zeros([4,2])
            body_keypoints = np.zeros([25,3])

        # store detection information
        imgnames_.append(imgname)
        widths_.append(W)
        heights_.append(H)
        scales_.append(scale)
        centers_.append(center)
        body_keypoints_.append(body_keypoints)
        seqnames_.append(seqname)
        valid_.append(valid)
        shots_.append(shot)

        # store body initialization
        pred_pose_.append(pred_pose_aa)
        pred_betas_.append(betas)
        cam_trans_.append(cam_trans)
        init_torso_.append(torso_joints)

    np.savez(args.output_npz, pred_aa=pred_pose_,
                              pred_betas=pred_betas_,
                              cam_trans=cam_trans_,
                              pred_torso=init_torso_,
                              imgname=imgnames_,
                              center=centers_,
                              scale=scales_,
                              width=widths_,
                              height=heights_,
                              seqname=seqnames_,
                              valid=valid_,
                              shot=shots_,
                              body_keypoints=body_keypoints_)

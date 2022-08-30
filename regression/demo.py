import os
import cv2
import smplx
import argparse
import numpy as np
from tqdm import tqdm
from yacs.config import CfgNode

import torch
from torch.utils.data import DataLoader

from regression.models import SMPL
from regression.datasets import DemoDataset
from regression.utils import Renderer
from regression.models import create_model, create_temporal_model

cfg = CfgNode(new_allowed=True)

# Define command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--npz', help='preprocessed npz file')
parser.add_argument('--output_folder', help='output folder')
parser.add_argument('--demo_cfg', default='regression/configs/demo.yaml', help='demo config file')

def run_demo(hmr_model, thmmr_model, cfg, dataset, output_folder):
    """Run evaluation on the datasets and metrics we report in the paper. """

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Transfer model to GPU
    hmr_model.to(device)
    thmmr_model.to(device)

    # Load SMPL model
    smpl = SMPL(cfg.SMPL.MODEL_PATH,
                batch_size=13*cfg.TRAIN.BATCH_SIZE,
                joint_regressor_extra=cfg.SMPL.JOINT_REGRESSOR_EXTRA).to(device)

    # Create dataloader for the dataset
    data_loader = DataLoader(dataset, batch_size=cfg.TRAIN.BATCH_SIZE, num_workers=cfg.GENERAL.NUM_WORKERS)

    renderer = Renderer(cfg, faces=smpl.faces)

    # Iterate over the entire dataset
    counter = 0
    for step, batch in enumerate(tqdm(data_loader, desc='Demo', total=len(data_loader))):

        # run the model
        images = batch['img'].to(device)
        src_key_padding_mask = batch['src_key_padding_mask'].to(device)
        batch_size_temp, seqlen, channels, imsize = images.shape[:4]
        curr_batch_size = batch_size_temp*seqlen
        images = images.view(curr_batch_size, channels, imsize, imsize)
        hmr_feats = hmr_model.backbone(images).max(-1)[0].max(-1)[0]
        hmr_feats = hmr_feats.view(batch_size_temp, seqlen, hmr_feats.size(-1))
        thmmr_feats = thmmr_model.encoder(hmr_feats, src_key_padding_mask)
        thmmr_feats = thmmr_feats.contiguous().view(-1, thmmr_feats.size(-1))
        pred_smpl_params, pred_camera, pred_smpl_params_list = thmmr_model.smpl_head(thmmr_feats)
        if batch_size_temp != cfg.TRAIN.BATCH_SIZE:
            smpl = SMPL(cfg.SMPL.MODEL_PATH,
                        batch_size=curr_batch_size,
                        joint_regressor_extra=cfg.SMPL.JOINT_REGRESSOR_EXTRA).to(device)
        pred_output = smpl(**pred_smpl_params, pose2rot=False)

        # visualization
        images = batch['img'][:,6]
        images = images * torch.tensor([0.229, 0.224, 0.225], device=images.device).reshape(1,3,1,1)
        images = images + torch.tensor([0.485, 0.456, 0.406], device=images.device).reshape(1,3,1,1)
        images = images.cpu().numpy()

        pred_vertices = pred_output.vertices[6::13].detach().cpu().numpy()

        dtype = pred_camera.dtype 
        pred_camera = pred_camera[6::13]
        camera_translation = torch.stack([pred_camera[:, 1],
                                          pred_camera[:, 2],
                                          2 * cfg.EXTRA.FOCAL_LENGTH / (pred_camera[:, 0] * torch.tensor(cfg.MODEL.IMAGE_SIZE, dtype=dtype, device=device) + 1e-9)], dim=-1).detach().cpu().numpy()
        focal_length = cfg.EXTRA.FOCAL_LENGTH / cfg.MODEL.IMAGE_SIZE * np.ones([curr_batch_size, 2])

        for bi in range(batch_size_temp):
            predictions = renderer.visualize(pred_vertices[[bi]], camera_translation[[bi]], images[[bi]], focal_length=focal_length[[bi]])
            cv2.imwrite(os.path.join(output_folder, '%04d.jpg' % counter), 255*np.transpose(predictions.numpy()[::-1], (1,2,0)))
            counter = counter + 1

if __name__ == '__main__':
    args = parser.parse_args()

    cfg = CfgNode(new_allowed=True)
    cfg.merge_from_file(args.demo_cfg)
    
    # load HMR model
    hmr_model = create_model(cfg)
    hmr_checkpoint = torch.load(cfg.MODEL.HMR_CHECKPOINT)
    hmr_model.load_state_dict(hmr_checkpoint['model'], strict=False)
    hmr_model.eval()
    
    # load tHMMR model
    thmmr_model = create_temporal_model(cfg)
    thmmr_checkpoint = torch.load(cfg.MODEL.THMMR_CHECKPOINT)
    thmmr_model.load_state_dict(thmmr_checkpoint['model'], strict=False)
    thmmr_model.eval()

    # Setup demo dataset
    dataset = DemoDataset(cfg, dataset_file=args.npz)

    # Run demo
    run_demo(hmr_model, thmmr_model, cfg, dataset, args.output_folder)

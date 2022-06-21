import os
import argparse
import numpy as np
import csv
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', default='mshot_data/datasets/ava', help='Path to store the AVA dataset')
parser.add_argument('--download', default=False, action='store_true', help='Download the AVA video files')
parser.add_argument('--extract_midframes', default=False, action='store_true', help='Extract the raw frames at 1 fps')
parser.add_argument('--extract_allframes', default=False, action='store_true', help='Extract the raw frames at 25 fps')

if __name__ == '__main__':
    args = parser.parse_args()

    # file with the names of the dataset
    ava_txt = 'ava_file_names_trainval_v2.1.txt'
    ava_txt_path = os.path.join(args.dataset_path, ava_txt)
    if not os.path.exists(ava_txt_path):
        os.system('wget https://s3.amazonaws.com/ava-dataset/annotations/%s -P %s' % (ava_txt, args.dataset_path))
    file_names = list(np.loadtxt(ava_txt_path, dtype='str'))

    # train/val filenames
    ava_zip = 'ava_v2.2'
    ava_zip_path = os.path.join(args.dataset_path, ava_zip)
    if not os.path.exists(ava_zip_path):
        os.system('wget https://research.google.com/ava/download/%s.zip -P %s' % (ava_zip, args.dataset_path))
        os.system('unzip %s.zip -d %s' % (ava_zip_path, ava_zip_path))
        os.system('rm %s.zip' % ava_zip_path)

    train_names = []
    csvreader = csv.reader(open(os.path.join(ava_zip_path, 'ava_train_v2.2.csv'), 'r'))
    for row in csvreader:
        train_names.append(row[0])
    train_names = np.unique(train_names)
    
    val_names = []
    csvreader = csv.reader(open(os.path.join(ava_zip_path, 'ava_val_v2.2.csv'), 'r'))
    for row in csvreader:
        val_names.append(row[0])
    val_names = list(np.unique(val_names))

    # download the data
    if args.download:
        for file_i in file_names:
            if file_i[:-4] in train_names:
                out_path = os.path.join(args.dataset_path, 'train', 'videos')
            elif file_i[:-4] in val_names:
                out_path = os.path.join(args.dataset_path, 'val', 'videos')
            os.system('wget https://s3.amazonaws.com/ava-dataset/trainval/%s -P %s' % (file_i, out_path))
   
    # extract the raw frames
    for file_i in file_names:
        
        # train or val set
        base_name = file_i.split('.')[0]
        if base_name in train_names:
            trainval = 'train'
        elif base_name in val_names:
            trainval = 'val'
        
        # video file
        file_path = os.path.join(args.dataset_path, trainval, 'videos', file_i)
    
        # create the folders to store the frames
        if args.extract_midframes:
            midframes_path = os.path.join(args.dataset_path, trainval, 'midframes', base_name)
            os.makedirs(midframes_path, exist_ok=True) 
        if args.extract_allframes:
            allframes_path = os.path.join(args.dataset_path, trainval, 'allframes', base_name)
            os.makedirs(allframes_path, exist_ok=True)
    
        # extract the frames
        for sec in range(902, 1799):
            if args.extract_midframes:
                frame_name = os.path.join(midframes_path, '%04d.jpg' % sec)
                ffmpeg_command = 'ffmpeg -ss %2f -i %s -frames:v 1 %s' % (sec, file_path, frame_name)
                subprocess.call(ffmpeg_command, shell=True)
            if args.extract_allframes:
                for msec in range(25):
                    t = sec + msec/25.
                    frame_name = os.path.join(allframes_path, '%04d_%02d.jpg' % (sec, msec))
                    ffmpeg_command = 'ffmpeg -ss %2f -i %s -frames:v 1 %s' % (t, file_path, frame_name)
                    subprocess.call(ffmpeg_command, shell=True)

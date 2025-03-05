import os
import os.path as osp
import sys
import cv2
import glob
import math
import torch
import gzip
import copy
import time
import json
import pickle
import base64
import imageio
import hashlib
import requests
import binascii
import numpy as np
from io import BytesIO
import urllib.request
import torch.nn.functional as F
import torchvision.utils as tvutils
from einops import rearrange
from PIL import Image, ImageDraw, ImageFont
from IPython import embed
import utils.transforms as data
from utils.video_op import *


def load_video_frames(ref_image_path, pose_file_path, pose_trans, resolution=[512, 768], max_frames=32, frame_interval = 2):
    
    for _ in range(5):
        try:
            dwpose_all = {}
            frames_all = {}
            for ii_index in sorted(os.listdir(pose_file_path)):
                if ii_index != "ref_pose.jpg":
                    dwpose_all[ii_index] = Image.open(pose_file_path+"/"+ii_index)
                    frames_all[ii_index] = Image.fromarray(cv2.cvtColor(cv2.imread(ref_image_path),cv2.COLOR_BGR2RGB)) 
                    # frames_all[ii_index] = Image.open(ref_image_path)
            
            pose_ref = Image.open(os.path.join(pose_file_path, "ref_pose.jpg"))

            # sample max_frames poses for video generation
            stride = frame_interval
            start_frame = 0 
            end_frame = (stride * (max_frames - 1) + 1)
            
            frame_list = []
            dwpose_list = []
            
            for i_index in range(start_frame, end_frame, stride):
                i_key = list(frames_all.keys())[i_index]
                i_frame = frames_all[i_key]
                if i_frame.mode != 'RGB':
                    i_frame = i_frame.convert('RGB')
                i_dwpose = dwpose_all[i_key]
                if i_dwpose.mode != 'RGB':
                    i_dwpose = i_dwpose.convert('RGB')
                frame_list.append(i_frame)
                dwpose_list.append(i_dwpose)
            have_frames = len(frame_list)>0
            dwpose_data = torch.zeros(max_frames, 3, resolution[1], resolution[0])
            misc_data = torch.zeros(max_frames, 3, resolution[1], resolution[0])
            if have_frames:
                misc_data_tmp = torch.stack([pose_trans(ss) for ss in frame_list], dim=0)
                dwpose_data_tmp = torch.stack([pose_trans(ss) for ss in dwpose_list], dim=0)
                misc_data[:len(frame_list), ...] = misc_data_tmp
                dwpose_data[:len(frame_list), ...] = dwpose_data_tmp            
            break
            
        except Exception as e:
            continue

    return misc_data, dwpose_data

if __name__ == '__main__':
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    resolution = [512, 768]

    video_trans = data.Compose([
        data.CenterCropWide(resolution),
        data.ToTensor(),
        data.Normalize(mean=mean, std=std)
        ]
    )
    pose_trans = data.Compose([
            data.Resize([512, 768]),
            data.ToTensor(),
            ]
    )
    for id in range(1, 8):
        local_path = f"./output/new_demo{id}.mp4"
        video_path = f'./output/demo{id}.mp4'
        ori_video_path = f'./output/ori_demo{id}.mp4'

        ref_image_path = f'data/images/demo{id}.jpg'
        pose_file_path = f'data/saved_pose/demo{id}'
        video_data = read_video_to_tensor(video_path, video_trans)
        ori_video_data = read_video_to_tensor(ori_video_path)

        source_img, dwpose_data = load_video_frames(ref_image_path, pose_file_path, pose_trans, resolution, 32, 2)
        source_img = source_img.unsqueeze(0)
        dwpose_data = dwpose_data.unsqueeze(0)

        bs_vd_local, frames_num = source_img.shape[:2]
        image_local = source_img[:,:1].clone().repeat(1,frames_num,1,1,1)
        image_local = rearrange(image_local, 'b f c h w -> b c f h w', b = bs_vd_local)

        bs_vd_local = dwpose_data.shape[0]
        dwpose_data = rearrange(dwpose_data, 'b f c h w -> b c f h w', b = bs_vd_local)



        model_kwargs_one_vis = [{
            'local_image': image_local,
            # 'dwpose': dwpose_data
            'ori_video': ori_video_data
        }]

        save_video_multiple_conditions_not_gif_horizontal_3col(local_path, video_data, model_kwargs_one_vis, source_img, mean, std, nrow=1, save_fps=8)

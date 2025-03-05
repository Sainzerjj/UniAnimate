import os
import sys
import os.path as osp
sys.path.insert(0, '/'.join(osp.realpath(__file__).split('/')[:-4]))
import cv2
import json
import torch
import random
import logging
import tempfile
import numpy as np
from copy import copy
from PIL import Image
from torch.utils.data import Dataset
from utils.registry_class import DATASETS
from IPython import embed
import traceback
import pickle

import utils.transforms as data
import torchvision.transforms as T

@DATASETS.register_class()
class UniAnimateDataset(Dataset):
    def __init__(self, 
            data_list,
            data_dir_list,
            max_words=1000,
            resolution=(384, 256),
            vit_resolution=(224, 224),
            max_frames=16,
            sample_fps=8,
            transforms=None,
            vit_transforms=None,
            pose_transforms=None,
            mask_transforms=None,
            get_first_frame=False,
            get_random_frame=False,
            **kwargs):

        self.max_words = max_words
        self.max_frames = max_frames
        self.resolution = resolution
        self.vit_resolution = vit_resolution
        self.sample_fps = sample_fps
        self.transforms = transforms
        self.vit_transforms = vit_transforms
        self.pose_transforms = pose_transforms
        self.mask_transforms = mask_transforms
        self.get_first_frame = get_first_frame
        self.get_random_frame = get_random_frame
        
        video_list = []
        for data_meta_path in data_list:
            video_list.extend(json.load(open(data_meta_path, "r")))
        self.vid_meta = video_list
        self.video_list = video_list



    def __getitem__(self, index):
        video_list = self.video_list[index]
        video_path = video_list["video_path"]
        pose_path = video_list["kps_path"]
        pose_mask_path = pose_path[:-4] + "_pose.pkl"
        region = video_list["region"]
        try:
            vit_frame, video_data, hand_masks_data, misc_data, dwpose_data, random_ref_frame_data, random_ref_dwpose_data = self.load_video_frames(pose_path, video_path, pose_mask_path, region)
        except Exception as e:
            logging.info('{} get frames failed... with error: {}'.format(video_path, e))
            raise e    
        return vit_frame, video_data, hand_masks_data, misc_data, dwpose_data, random_ref_frame_data, random_ref_dwpose_data

    
    def draw_handmasks(self, pose, H, W, thresh=0.3):
        canvas = np.zeros((H, W), dtype=np.uint8)
        if "hands" not in pose:
            return canvas
        for peaks, score in zip(pose["hands"], pose["hands_score"]):
            if not np.all(score > thresh):
                continue
            x1, y1 = np.min(peaks, 0) * [W, H]
            x2, y2 = np.max(peaks, 0) * [W, H]
            x1 = int(max(0, x1 - 24))
            y1 = int(max(0, y1 - 24))
            x2 = int(x2 + 24)
            y2 = int(y2 + 24)
            canvas[y1:y2, x1:x2] = 255
        return canvas

    def calculate_bounding_box_all_frames(self, pose_meta_data, H, W, indices=[[2, 5, 8, 11], [2, 5, 8, 11], [2, 5, 8, 11]]):  
        all_keypoints = []  
        indices = random.choices(indices, [0.4, 0.3, 0.3])[0]
        
        for frame in pose_meta_data:  
            candidate = frame['bodies']['candidate'] 
            for index in indices:  
                if index < len(candidate):  
                    all_keypoints.append(candidate[index])  
        if not all_keypoints:  
            return None  
        xs, ys = zip(*[(point[0] * W, point[1] * H) for point in all_keypoints])  
        x_min, x_max = max(0, int(min(xs)-40)), min(W, int(max(xs)+40))  
        y_min, y_max = max(0, int(min(ys)-60)), min(H, int(max(ys)+60))  
        return (x_min, y_min, x_max, y_max)
    
    def crop_frames_poses(self, frame_list, dwpose_list, hand_mask_list, random_ref_frame, random_ref_dwpose, bbox):  
        if not frame_list or not dwpose_list:  
            raise ValueError("Frame list and DW pose list must not be empty.")  
        
        dwpose_width, dwpose_height = dwpose_list[0].size  
        frame_width, frame_height = frame_list[0].size  
        x_min, y_min, x_max, y_max = bbox  
        
        scale_x = dwpose_width / frame_width 
        scale_y = dwpose_height / frame_height    
        
        x_min_dwpose = int(x_min * scale_x)  
        y_min_dwpose = int(y_min * scale_y)  
        x_max_dwpose = int(x_max * scale_x)  
        y_max_dwpose = int(y_max * scale_y)  

        # 转换为张量  
        frame_tensors = [T.ToTensor()(frame) for frame in frame_list]  
        dwpose_tensors = [T.ToTensor()(dwpose) for dwpose in dwpose_list]  
        hand_mask_tensors = [T.ToTensor()(hand_mask) for hand_mask in hand_mask_list]
        random_ref_frame_tensors = T.ToTensor()(random_ref_frame)
        random_ref_dwpose_tensors = T.ToTensor()(random_ref_dwpose)

        # 将列表转换为张量  
        frame_tensor_stack = torch.stack(frame_tensors)  
        dwpose_tensor_stack = torch.stack(dwpose_tensors)  
        hand_mask_tensor_stack = torch.stack(hand_mask_tensors)

        # 执行裁剪操作  
        cropped_frame_tensor = frame_tensor_stack[:, :, y_min:y_max, x_min:x_max]  
        cropped_dwpose_tensor = dwpose_tensor_stack[:, :, y_min_dwpose:y_max_dwpose, x_min_dwpose:x_max_dwpose]  
        cropped_hand_mask_tensor = hand_mask_tensor_stack[:, :, y_min:y_max, x_min:x_max]
        cropped_random_ref_frame = random_ref_frame_tensors[:, y_min:y_max, x_min:x_max]
        cropped_random_ref_dwpose = random_ref_dwpose_tensors[:, y_min_dwpose:y_max_dwpose, x_min_dwpose:x_max_dwpose]

        # 转换回 PIL 图像  
        cropped_frame_list = [T.ToPILImage()(img) for img in cropped_frame_tensor]  
        cropped_dwpose_list = [T.ToPILImage()(img) for img in cropped_dwpose_tensor]  
        cropped_hand_mask_list = [T.ToPILImage()(img) for img in cropped_hand_mask_tensor]
        cropped_random_ref_frame = T.ToPILImage()(cropped_random_ref_frame)
        cropped_random_ref_dwpose = T.ToPILImage()(cropped_random_ref_dwpose)
        # print(cropped_frame_list[0].size)

        return cropped_frame_list, cropped_dwpose_list, cropped_hand_mask_list, cropped_random_ref_frame, cropped_random_ref_dwpose


    def load_video_frames(self, pose_path, video_path, pose_mask_path, region):
        for _ in range(5):
            # sample max_frames poses for video generation
            with open(pose_mask_path, "rb") as f:
                poses = pickle.load(f)
            capture_video = cv2.VideoCapture(video_path)
            capture_pose = cv2.VideoCapture(pose_path)
            _fps = capture_video.get(cv2.CAP_PROP_FPS)
            _total_frame_num = int(capture_video.get(cv2.CAP_PROP_FRAME_COUNT))
            video_width = int(capture_video.get(cv2.CAP_PROP_FRAME_WIDTH))  
            video_height = int(capture_video.get(cv2.CAP_PROP_FRAME_HEIGHT))  
            assert _total_frame_num==len(poses)
            frame_list = []
            dwpose_list = []
            hand_mask_list = []
            # print("sample_fps:", self.sample_fps)
            try:
                dwpose_all = {}
                frames_all = {}
                for index in range(0, _total_frame_num):
                    ii_index = f"{index:04d}" 
                    success, i_frame_pose = capture_pose.read()
                    if (not success) or (i_frame_pose is None):
                        print(f"Unable to read the frame {ii_index} of the video from {pose_path}")
                        break
                    success, i_frame_video = capture_video.read()
                    if (not success) or (i_frame_video is None):
                        print(f"Unable to read the frame {ii_index} of the video from {video_path}")
                        break
                    dwpose_all[ii_index] = Image.fromarray(cv2.cvtColor(i_frame_pose, cv2.COLOR_BGR2RGB))
                    frames_all[ii_index] = Image.fromarray(cv2.cvtColor(i_frame_video, cv2.COLOR_BGR2RGB)) 
                    # frames_all[ii_index] = Image.open(ref_image_path)
                if (not success) or (i_frame_video is None):
                    print(f"Skip the video in {video_path} !!!")
                    break 
                # stride = round(_fps / self.sample_fps)
                if _total_frame_num <= self.max_frames:
                    stride = 1
                else:
                    stride = max(round(_fps / self.sample_fps), 1)
                # cover_frame_num = (stride * self.max_frames)
                cover_frame_num = stride * self.max_frames
                if _total_frame_num < cover_frame_num + 5:
                    print('_total_frame_num is smaller than cover_frame_num, the sampled frame interval is changed')
                    start_frame = 0   # we set start_frame = 0 because the pose alignment is performed on the first frame
                    end_frame = _total_frame_num
                else:
                    start_frame = random.randint(0, _total_frame_num - cover_frame_num - 5)  # we set start_frame = 0 because the pose alignment is performed on the first frame
                    end_frame = start_frame + cover_frame_num
                
                # rand_index = 0
                rand_index = random.randint(0, _total_frame_num - 1)
                random_ref_frame = frames_all[list(frames_all.keys())[rand_index]]
                if random_ref_frame.mode != 'RGB':
                    random_ref_frame = random_ref_frame.convert('RGB')
                # random_ref_dwpose = pose_ref 
                random_ref_dwpose = dwpose_all[list(frames_all.keys())[rand_index]] 
                if random_ref_dwpose.mode != 'RGB':
                    random_ref_dwpose = random_ref_dwpose.convert('RGB')
                for i_index in range(start_frame, end_frame, stride):
                    # print(start_frame, end_frame, self.sample_fps, stride, len(list(frames_all.keys())), i_index)
                    i_key = list(frames_all.keys())[i_index]
                    hand_masks = Image.fromarray(self.draw_handmasks(
                        poses[i_index], video_height, video_width
                    ))  
                    # capture.set(cv2.CAP_PROP_POS_FRAMES, i_index)
                    # success, i_frame = capture.read()
                    # if not success:
                    #     print(f"Unable to read the frame {i_index} of the video from {video_path}")
                    # i_frame = Image.fromarray(cv2.cvtColor(i_frame, cv2.COLOR_BGR2RGB))  
                    i_frame = frames_all[i_key]          
                    if i_frame.mode != 'RGB':
                        i_frame = i_frame.convert('RGB')
                    i_dwpose = dwpose_all[i_key]
                    if i_dwpose.mode != 'RGB':
                        i_dwpose = i_dwpose.convert('RGB')
                    frame_list.append(i_frame)
                    dwpose_list.append(i_dwpose)
                    hand_mask_list.append(hand_masks)
                if region:
                    bbox = self.calculate_bounding_box_all_frames(poses, H=video_height, W=video_width, indices=[[2, 5, 4, 7, 14, 15], [3, 6, 4, 7, 8, 11], [4, 7, 10, 13]])
                    frame_list, dwpose_list, hand_mask_list, random_ref_frame, random_ref_dwpose = self.crop_frames_poses(frame_list, dwpose_list, hand_mask_list, random_ref_frame, random_ref_dwpose, bbox)
                break  
            except Exception as e:
                logging.info('{} read video frame failed with error: {}'.format(pose_path, e))
                traceback.print_tb(e.__traceback__)
                continue
        
        vit_frame = torch.zeros(3, self.vit_resolution[1], self.vit_resolution[0])
        video_data = torch.zeros(self.max_frames, 3, self.resolution[1], self.resolution[0])
        dwpose_data = torch.zeros(self.max_frames, 3, self.resolution[1], self.resolution[0])
        hand_masks_data = torch.zeros(self.max_frames, 1, self.resolution[1], self.resolution[0])
        misc_data = torch.zeros(self.max_frames, 3, self.resolution[1], self.resolution[0])
        random_ref_frame_data = torch.zeros(self.max_frames, 3, self.resolution[1], self.resolution[0]) 
        random_ref_dwpose_data = torch.zeros(self.max_frames, 3, self.resolution[1], self.resolution[0])

        try:
            if self.get_first_frame:
                ref_idx = 0
            elif self.get_random_frame:
                ref_idx = np.random.randint(0, len(frame_list))
            else:
                ref_idx = int(len(frame_list)/2)
            while len(frame_list) > self.max_frames:
                del frame_list[-1]
                del dwpose_list[-1]
                del hand_mask_list[-1]
            middle_indix = ref_idx
            have_frames = len(frame_list) > 0
            if have_frames:
                ref_frame = frame_list[middle_indix]
                vit_frame = self.vit_transforms(ref_frame)
                video_data[:len(frame_list), ...] = torch.stack([self.transforms(ss) for ss in frame_list], dim=0) 
                hand_masks_data[:len(frame_list), ...] = torch.stack([self.mask_transforms(ss) for ss in hand_mask_list], dim=0)
                misc_data[:len(frame_list), ...] = torch.stack([self.pose_transforms(ss) for ss in frame_list], dim=0)
                dwpose_data[:len(frame_list), ...] = torch.stack([self.pose_transforms(ss) for ss in dwpose_list], dim=0)
                random_ref_frame_data[:,...] = self.pose_transforms(random_ref_frame)
                random_ref_dwpose_data[:,...] = self.pose_transforms(random_ref_dwpose) 
        except Exception as e:
            logging.info('Error: {}'.format(e))
            traceback.print_tb(e.__traceback__)
                
        return vit_frame, video_data, hand_masks_data, misc_data, dwpose_data, random_ref_frame_data, random_ref_dwpose_data

    def __len__(self):
        return len(self.video_list)



def main():
    train_trans = data.Compose([
        data.Resize((768, 1216)),
        data.ToTensor(),
        data.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
        ])

    train_trans_pose = data.Compose([
        data.Resize((768, 1216)),
        data.ToTensor(),
        ]
    )

    train_trans_mask = data.Compose([
        data.Resize((768, 1216)),
        data.ToTensor(),
        data.Normalize(mean=0.5, std=0.5)
        ])

    train_trans_vit = T.Compose([
                data.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))])
    
    vid_dataset={
        'type': 'UniAnimateDataset',
        # 'data_list': ['/home/admin/workspace/aop_lab/xianghaodong/0-dataset/animate_data/TikTok/vid_list.txt', '/home/admin/workspace/aop_lab/xianghaodong/0-dataset/animate_data/Fashion/vid_list.txt'],
        # 'data_list': ['/home/admin/workspace/aop_lab/zhoushengzhe/datasets/huiwa_video/train_data/vid_list.txt', ],
        # 'data_dir_list': ['/home/admin/workspace/aop_lab/xianghaodong/0-dataset/animate_data/TikTok/videos', '/home/admin/workspace/aop_lab/xianghaodong/0-dataset/animate_data/Fashion/videos'],
        'data_dir_list': ['/home/admin/workspace/aop_lab/zhoushengzhe/datasets/huiwa_video/train_data/videos', ],
        'vit_resolution': [224, 224],
        'resolution': [768, 1216], # [768, 1216],
        'get_first_frame': True,
        'max_words': 1000,
        'data_list': [
            "moore_meta_data_qy/new_dance_dataset.json",
            "moore_meta_data_qy/our_dataset_8.json",
        ]
    }
    dataset = DATASETS.build(vid_dataset, sample_fps=2, transforms=train_trans, vit_transforms=train_trans_vit, pose_transforms=train_trans_pose, mask_transforms=train_trans_mask, max_frames=32)
    print(len(dataset))
    while True:
        sample = dataset[np.random.randint(0, len(dataset)-1)]
    
if __name__ == '__main__':
    main()
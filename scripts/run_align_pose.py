# Openpose
# Original from CMU https://github.com/CMU-Perceptual-Computing-Lab/openpose
# 2nd Edited by https://github.com/Hzzone/pytorch-openpose
# 3rd Edited by ControlNet
# 4th Edited by ControlNet (added face and correct hands)

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import cv2
import torch
import numpy as np
import json
import copy
import torch
import random
import argparse
import shutil
import tempfile
import subprocess
import numpy as np
import math

import torch.multiprocessing as mp
import torch.distributed as dist
import pickle
import logging
from io import BytesIO
import oss2 as oss
import os.path as osp

import sys
import dwpose.util as util
from dwpose.wholebody import Wholebody

import collections
import copy 

def draw_points(points, resume=True, point_color=(0, 0, 255), point_radius=3, save_path="/nas_mount/xianghaodong/pose_points.jpg", window_size=(512, 768)):
    W, H = window_size[0], window_size[1]
    points = [[p[0] * W, p[1] * H] for p in points]

    if os.path.exists(save_path) and resume:  
        img = cv2.imread(save_path)  
        img = cv2.resize(img, window_size)  
    else:  
        img = np.zeros((H, W, 3), dtype=np.uint8)  

    for idx, point in enumerate(points):  
        point = tuple(map(int, point))  
        cv2.circle(img, point, point_radius, point_color, -1)  
        cv2.putText(img, str(idx), (point[0] + 5, point[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imwrite(save_path, img)
    


def smoothing_factor(t_e, cutoff):
    r = 2 * math.pi * cutoff * t_e
    return r / (r + 1)


def exponential_smoothing(a, x, x_prev):
    return a * x + (1 - a) * x_prev

# not used
class OneEuroFilter:
    def __init__(self, t0, x0, dx0=0.0, min_cutoff=1.0, beta=0.0,
                 d_cutoff=1.0):
        """Initialize the one euro filter."""
        # The parameters.
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        # Previous values.
        self.x_prev = x0
        self.dx_prev = float(dx0)
        self.t_prev = float(t0)

    def __call__(self, t, x):
        """Compute the filtered signal."""
        t_e = t - self.t_prev

        # The filtered derivative of the signal.
        a_d = smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = exponential_smoothing(a_d, dx, self.dx_prev)

        # The filtered signal.
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = smoothing_factor(t_e, cutoff)
        x_hat = exponential_smoothing(a, x, self.x_prev)

        # Memorize the previous values.
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t

        return x_hat


def get_logger(name="essmc2"):
    logger = logging.getLogger(name)
    logger.propagate = False
    if len(logger.handlers) == 0:
        std_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        std_handler.setFormatter(formatter)
        std_handler.setLevel(logging.INFO)
        logger.setLevel(logging.INFO)
        logger.addHandler(std_handler)
    return logger

class DWposeDetector:
    def __init__(self):

        self.pose_estimation = Wholebody() # return keypoints, scores

    def __call__(self, oriImg):
        oriImg = oriImg.copy()
        H, W, C = oriImg.shape
        with torch.no_grad():
            candidate, subset = self.pose_estimation(oriImg)
            candidate = candidate[0][np.newaxis, :, :]
            subset = subset[0][np.newaxis, :]
            nums, keys, locs = candidate.shape
            candidate[..., 0] /= float(W)
            candidate[..., 1] /= float(H)
            body = candidate[:,:18].copy()
            body = body.reshape(nums*18, locs)
            score = subset[:,:18].copy()
            
            for i in range(len(score)):
                for j in range(len(score[i])):
                    if score[i][j] > 0.3:
                        score[i][j] = int(18*i+j)
                    else:
                        score[i][j] = -1

            un_visible = subset<0.3
            candidate[un_visible] = -1

            bodyfoot_score = subset[:,:24].copy()
            for i in range(len(bodyfoot_score)):
                for j in range(len(bodyfoot_score[i])):
                    if bodyfoot_score[i][j] > 0.3:
                        bodyfoot_score[i][j] = int(18*i+j)
                    else:
                        bodyfoot_score[i][j] = -1
            if -1 not in bodyfoot_score[:,18] and -1 not in bodyfoot_score[:,19]:
                bodyfoot_score[:,18] = np.array([18.]) 
            else:
                bodyfoot_score[:,18] = np.array([-1.])
            if -1 not in bodyfoot_score[:,21] and -1 not in bodyfoot_score[:,22]:
                bodyfoot_score[:,19] = np.array([19.]) 
            else:
                bodyfoot_score[:,19] = np.array([-1.])
            bodyfoot_score = bodyfoot_score[:, :20]

            bodyfoot = candidate[:,:24].copy()
            
            for i in range(nums):
                if -1 not in bodyfoot[i][18] and -1 not in bodyfoot[i][19]:
                    bodyfoot[i][18] = (bodyfoot[i][18]+bodyfoot[i][19])/2
                else:
                    bodyfoot[i][18] = np.array([-1., -1.])
                if -1 not in bodyfoot[i][21] and -1 not in bodyfoot[i][22]:
                    bodyfoot[i][19] = (bodyfoot[i][21]+bodyfoot[i][22])/2
                else:
                    bodyfoot[i][19] = np.array([-1., -1.])
            
            bodyfoot = bodyfoot[:,:20,:]
            bodyfoot = bodyfoot.reshape(nums*20, locs)

            foot = candidate[:,18:24]

            faces = candidate[:,24:92]

            hands = candidate[:,92:113]
            hands = np.vstack([hands, candidate[:,113:]])
            
            # bodies = dict(candidate=body, subset=score)
            bodies = dict(candidate=bodyfoot, subset=bodyfoot_score)
            pose = dict(bodies=bodies, hands=hands, faces=faces)

            # return draw_pose(pose, H, W)
            return pose

def draw_pose(pose, H, W):
    bodies = pose['bodies']
    faces = pose['faces']
    hands = pose['hands']
    candidate = bodies['candidate']
    subset = bodies['subset']
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)

    canvas = util.draw_body_and_foot(canvas, candidate, subset)

    canvas = util.draw_handpose(canvas, hands)

    canvas_without_face = copy.deepcopy(canvas)

    canvas = util.draw_facepose(canvas, faces)

    return canvas_without_face, canvas

def dw_func(_id, frame, dwpose_model, dwpose_woface_folder='tmp_dwpose_wo_face', dwpose_withface_folder='tmp_dwpose_with_face'):
    
    # frame = cv2.imread(frame_name, cv2.IMREAD_COLOR)
    pose = dwpose_model(frame)

    return pose

def mp_main(args):
    
    if args.source_video_paths.endswith('mp4'):
        video_paths = [args.source_video_paths]
    else:
        # video list
        video_paths = [os.path.join(args.source_video_paths, frame_name) for frame_name in os.listdir(args.source_video_paths)]

    
    logger.info("There are {} videos for extracting poses".format(len(video_paths)))

    logger.info('LOAD: DW Pose Model')
    dwpose_model = DWposeDetector()  
    
    
    for i, file_path in enumerate(video_paths):
        results_vis = []
        logger.info(f"{i}/{len(video_paths)}, {file_path}")
        videoCapture = cv2.VideoCapture(file_path)
        while videoCapture.isOpened():
            # get a frame
            ret, frame = videoCapture.read()
            if ret:
                pose = dw_func(i, frame, dwpose_model)
                results_vis.append(pose)
            else:
                break
        logger.info(f'all frames in {file_path} have been read.')
        videoCapture.release()

        # added
        # results_vis = results_vis[8:]
        print(len(results_vis))

        saved_pose_dir = os.path.join(args.saved_pose_dir, file_path.split('/')[-1][:-4])
        ref_name = args.ref_name
        save_motion = saved_pose_dir
        os.system(f'rm -rf {save_motion}');
        os.makedirs(save_motion, exist_ok=True)
        save_warp = saved_pose_dir
        # os.makedirs(save_warp, exist_ok=True)
        
        ref_frame = cv2.imread(ref_name, cv2.IMREAD_COLOR)
        pose_ref = dw_func(i, ref_frame, dwpose_model)

        std_frame_idx = args.std_frame_idx # 默认为driving pose第0帧 手动传参数修正
        if std_frame_idx == -1:
            # 选肩宽最大的一帧 做对比
            std_frame_idx = np.argmax([(np.abs(results_vis[i]['bodies']['candidate'][2][0] - results_vis[i]['bodies']['candidate'][5][0])) for i in range(len(results_vis))])
        print(f'[INFO] ====================== std_frame_idx: {std_frame_idx} ====================== ' )

        bodies = results_vis[std_frame_idx]['bodies']
        faces = results_vis[std_frame_idx]['faces']
        hands = results_vis[std_frame_idx]['hands']
        candidate = bodies['candidate']
        
        first_frame_pose = copy.deepcopy(results_vis[std_frame_idx])  
        # first_frame_pose = results_vis[std_frame_idx].copy()

        ref_bodies = pose_ref['bodies']
        ref_faces = pose_ref['faces']
        ref_hands = pose_ref['hands']
        ref_candidate = ref_bodies['candidate']


        ref_2_x = ref_candidate[2][0] # 右肩
        ref_2_y = ref_candidate[2][1]
        ref_5_x = ref_candidate[5][0] # 左肩
        ref_5_y = ref_candidate[5][1]
        ref_8_x = ref_candidate[8][0] # 右胯
        ref_8_y = ref_candidate[8][1]
        ref_11_x = ref_candidate[11][0] # 左胯
        ref_11_y = ref_candidate[11][1]
        ref_center1 = 0.5*(ref_candidate[2]+ref_candidate[5]) # 肩中心
        ref_center2 = 0.5*(ref_candidate[8]+ref_candidate[11]) # 胯中心

        zero_2_x = candidate[2][0]
        zero_2_y = candidate[2][1]
        zero_5_x = candidate[5][0]
        zero_5_y = candidate[5][1]
        zero_8_x = candidate[8][0]
        zero_8_y = candidate[8][1]
        zero_11_x = candidate[11][0]
        zero_11_y = candidate[11][1]
        zero_center1 = 0.5*(candidate[2]+candidate[5])
        zero_center2 = 0.5*(candidate[8]+candidate[11])

        # ============================ stage 1 ============================
        x_ratio = (ref_5_x-ref_2_x)/(zero_5_x-zero_2_x) # （ref左肩x - ref右肩x）/（左肩x - 右肩x）
        y_ratio = (ref_center2[1]-ref_center1[1])/(zero_center2[1]-zero_center1[1]) # （ref肩中心y - ref胯中心y）/(肩中心y - 胯中心y)

        len_ratio = (ref_center2[0]-ref_center1[0])/(zero_center2[0]-zero_center1[0])
        align_parameters = collections.defaultdict(dict)
        align_parameters['ratio'].update({'len_ratio' : x_ratio})

        results_vis[std_frame_idx]['bodies']['candidate'][:,0] *= x_ratio
        results_vis[std_frame_idx]['bodies']['candidate'][:,1] *= y_ratio
        results_vis[std_frame_idx]['faces'][:,:,0] *= x_ratio
        results_vis[std_frame_idx]['faces'][:,:,1] *= y_ratio
        results_vis[std_frame_idx]['hands'][:,:,0] *= x_ratio
        results_vis[std_frame_idx]['hands'][:,:,1] *= y_ratio

        # 检查pose_ref['bodies']['candidate']中消失的点 
        # 用对称点弥补? 如果pose比较奇怪呢？用对应向量弥补比较好
        # 其实无妨 只要出现就行 先简单点计算对称点
        left_bodies_idx = [14, 16, 2, 3, 4, 8, 9, 10, 19]
        right_bodies_idx = [15, 17, 5, 6, 7, 11, 12, 13, 18]
        
        def symmetric_point(c1, c2, p):  
            x1, y1 = c1  
            x2, y2 = c2  
            x, y = p  
            if x1 == x2:  # 对称轴垂直，即 x = x1  
                x_prime = 2 * x1 - x  
                y_prime = y  
            elif y1 == y2:  # 对称轴水平，即 y = y1  
                x_prime = x  
                y_prime = 2 * y1 - y  
            else:  
                m = (y2 - y1) / (x2 - x1)  
                b = y1 - m * x1  
                x_prime = ((1 - m**2) * x + 2 * m * (y - b)) / (m**2 + 1)  
                y_prime = ((m**2 - 1) * y + 2 * m * x + 2 * b) / (m**2 + 1)  
            return [x_prime, y_prime] 

        for i in range(len(left_bodies_idx)):
            # dwpose 未检测到的点为（-1， -1）
            if pose_ref['bodies']['candidate'][left_bodies_idx[i],0] < 0 and pose_ref['bodies']['candidate'][right_bodies_idx[i],0] > 0 :
                pose_ref['bodies']['candidate'][left_bodies_idx[i]] = symmetric_point(ref_center1, ref_center2, pose_ref['bodies']['candidate'][right_bodies_idx[i]])
            elif pose_ref['bodies']['candidate'][right_bodies_idx[i],0] < 0 and pose_ref['bodies']['candidate'][left_bodies_idx[i],0] > 0 :
                pose_ref['bodies']['candidate'][right_bodies_idx[i]] = symmetric_point(ref_center1, ref_center2, pose_ref['bodies']['candidate'][left_bodies_idx[i]])
            
        
        # ============================ stage 2 ============================
        ########neck######## 1 --- 0 14 15 16 17
        l_neck_ref = ((ref_candidate[0][0] - ref_candidate[1][0]) ** 2 + (ref_candidate[0][1] - ref_candidate[1][1]) ** 2) ** 0.5
        l_neck_0 = ((candidate[0][0] - candidate[1][0]) ** 2 + (candidate[0][1] - candidate[1][1]) ** 2) ** 0.5
        neck_ratio = l_neck_ref / l_neck_0

        x_offset_neck = (candidate[1][0]-candidate[0][0])*(1.-neck_ratio) 
        y_offset_neck = (candidate[1][1]-candidate[0][1])*(1.-neck_ratio) 

        align_parameters['ratio'].update({'neck_ratio' : neck_ratio})
        align_parameters['offset'].update({'offset_neck' : [x_offset_neck, y_offset_neck]})

        '''
        | 0------1
        
        | 0-->0---1

        | (1-ratio) ratio
        offset实现矢量起点不变的平移
        然后层层对齐下去
        '''

        results_vis[std_frame_idx]['bodies']['candidate'][0,0] += x_offset_neck
        results_vis[std_frame_idx]['bodies']['candidate'][0,1] += y_offset_neck
        results_vis[std_frame_idx]['bodies']['candidate'][14,0] += x_offset_neck
        results_vis[std_frame_idx]['bodies']['candidate'][14,1] += y_offset_neck
        results_vis[std_frame_idx]['bodies']['candidate'][15,0] += x_offset_neck
        results_vis[std_frame_idx]['bodies']['candidate'][15,1] += y_offset_neck
        results_vis[std_frame_idx]['bodies']['candidate'][16,0] += x_offset_neck
        results_vis[std_frame_idx]['bodies']['candidate'][16,1] += y_offset_neck
        results_vis[std_frame_idx]['bodies']['candidate'][17,0] += x_offset_neck
        results_vis[std_frame_idx]['bodies']['candidate'][17,1] += y_offset_neck
        
        ########shoulder2######## 1 --- 2 3 4 hands[1]
        l_shoulder2_ref = ((ref_candidate[2][0] - ref_candidate[1][0]) ** 2 + (ref_candidate[2][1] - ref_candidate[1][1]) ** 2) ** 0.5
        l_shoulder2_0 = ((candidate[2][0] - candidate[1][0]) ** 2 + (candidate[2][1] - candidate[1][1]) ** 2) ** 0.5

        shoulder2_ratio = l_shoulder2_ref / l_shoulder2_0

        x_offset_shoulder2 = (candidate[1][0]-candidate[2][0])*(1.-shoulder2_ratio)
        y_offset_shoulder2 = (candidate[1][1]-candidate[2][1])*(1.-shoulder2_ratio)

        align_parameters['ratio'].update({'shoulder2_ratio' : shoulder2_ratio})
        align_parameters['offset'].update({'offset_shoulder2' : [x_offset_shoulder2, y_offset_shoulder2]})

        results_vis[std_frame_idx]['bodies']['candidate'][2,0] += x_offset_shoulder2
        results_vis[std_frame_idx]['bodies']['candidate'][2,1] += y_offset_shoulder2
        results_vis[std_frame_idx]['bodies']['candidate'][3,0] += x_offset_shoulder2
        results_vis[std_frame_idx]['bodies']['candidate'][3,1] += y_offset_shoulder2
        results_vis[std_frame_idx]['bodies']['candidate'][4,0] += x_offset_shoulder2
        results_vis[std_frame_idx]['bodies']['candidate'][4,1] += y_offset_shoulder2
        results_vis[std_frame_idx]['hands'][1,:,0] += x_offset_shoulder2
        results_vis[std_frame_idx]['hands'][1,:,1] += y_offset_shoulder2

        ########shoulder5######## 1 --- 5 6 7 hands[0]
        l_shoulder5_ref = ((ref_candidate[5][0] - ref_candidate[1][0]) ** 2 + (ref_candidate[5][1] - ref_candidate[1][1]) ** 2) ** 0.5
        l_shoulder5_0 = ((candidate[5][0] - candidate[1][0]) ** 2 + (candidate[5][1] - candidate[1][1]) ** 2) ** 0.5

        shoulder5_ratio = l_shoulder5_ref / l_shoulder5_0

        x_offset_shoulder5 = (candidate[1][0]-candidate[5][0])*(1.-shoulder5_ratio)
        y_offset_shoulder5 = (candidate[1][1]-candidate[5][1])*(1.-shoulder5_ratio)

        align_parameters['ratio'].update({'shoulder5_ratio' : shoulder5_ratio})
        align_parameters['offset'].update({'offset_shoulder5' : [x_offset_shoulder5, y_offset_shoulder5]})

        results_vis[std_frame_idx]['bodies']['candidate'][5,0] += x_offset_shoulder5
        results_vis[std_frame_idx]['bodies']['candidate'][5,1] += y_offset_shoulder5
        results_vis[std_frame_idx]['bodies']['candidate'][6,0] += x_offset_shoulder5
        results_vis[std_frame_idx]['bodies']['candidate'][6,1] += y_offset_shoulder5
        results_vis[std_frame_idx]['bodies']['candidate'][7,0] += x_offset_shoulder5
        results_vis[std_frame_idx]['bodies']['candidate'][7,1] += y_offset_shoulder5
        results_vis[std_frame_idx]['hands'][0,:,0] += x_offset_shoulder5
        results_vis[std_frame_idx]['hands'][0,:,1] += y_offset_shoulder5

        ########arm3######## 2 --- 3 4 hands[1]
        l_arm3_ref = ((ref_candidate[3][0] - ref_candidate[2][0]) ** 2 + (ref_candidate[3][1] - ref_candidate[2][1]) ** 2) ** 0.5
        l_arm3_0 = ((candidate[3][0] - candidate[2][0]) ** 2 + (candidate[3][1] - candidate[2][1]) ** 2) ** 0.5

        arm3_ratio = l_arm3_ref / l_arm3_0

        x_offset_arm3 = (candidate[2][0]-candidate[3][0])*(1.-arm3_ratio)
        y_offset_arm3 = (candidate[2][1]-candidate[3][1])*(1.-arm3_ratio)

        align_parameters['ratio'].update({'arm3_ratio' : arm3_ratio})
        align_parameters['offset'].update({'offset_arm3' : [x_offset_arm3, y_offset_arm3]})

        results_vis[std_frame_idx]['bodies']['candidate'][3,0] += x_offset_arm3
        results_vis[std_frame_idx]['bodies']['candidate'][3,1] += y_offset_arm3
        results_vis[std_frame_idx]['bodies']['candidate'][4,0] += x_offset_arm3
        results_vis[std_frame_idx]['bodies']['candidate'][4,1] += y_offset_arm3
        results_vis[std_frame_idx]['hands'][1,:,0] += x_offset_arm3
        results_vis[std_frame_idx]['hands'][1,:,1] += y_offset_arm3

        ########arm4######## 3 --- 4 hands[1]
        l_arm4_ref = ((ref_candidate[4][0] - ref_candidate[3][0]) ** 2 + (ref_candidate[4][1] - ref_candidate[3][1]) ** 2) ** 0.5
        l_arm4_0 = ((candidate[4][0] - candidate[3][0]) ** 2 + (candidate[4][1] - candidate[3][1]) ** 2) ** 0.5

        arm4_ratio = l_arm4_ref / l_arm4_0

        x_offset_arm4 = (candidate[3][0]-candidate[4][0])*(1.-arm4_ratio)
        y_offset_arm4 = (candidate[3][1]-candidate[4][1])*(1.-arm4_ratio)

        align_parameters['ratio'].update({'arm4_ratio' : arm4_ratio})
        align_parameters['offset'].update({'offset_arm4' : [x_offset_arm4, y_offset_arm4]})

        results_vis[std_frame_idx]['bodies']['candidate'][4,0] += x_offset_arm4
        results_vis[std_frame_idx]['bodies']['candidate'][4,1] += y_offset_arm4
        results_vis[std_frame_idx]['hands'][1,:,0] += x_offset_arm4
        results_vis[std_frame_idx]['hands'][1,:,1] += y_offset_arm4

        ########arm6######## 5 --- 6 7 hands[0]
        l_arm6_ref = ((ref_candidate[6][0] - ref_candidate[5][0]) ** 2 + (ref_candidate[6][1] - ref_candidate[5][1]) ** 2) ** 0.5
        l_arm6_0 = ((candidate[6][0] - candidate[5][0]) ** 2 + (candidate[6][1] - candidate[5][1]) ** 2) ** 0.5

        arm6_ratio = l_arm6_ref / l_arm6_0

        x_offset_arm6 = (candidate[5][0]-candidate[6][0])*(1.-arm6_ratio)
        y_offset_arm6 = (candidate[5][1]-candidate[6][1])*(1.-arm6_ratio)

        align_parameters['ratio'].update({'arm6_ratio' : arm6_ratio})
        align_parameters['offset'].update({'offset_arm6' : [x_offset_arm6, y_offset_arm6]})

        results_vis[std_frame_idx]['bodies']['candidate'][6,0] += x_offset_arm6
        results_vis[std_frame_idx]['bodies']['candidate'][6,1] += y_offset_arm6
        results_vis[std_frame_idx]['bodies']['candidate'][7,0] += x_offset_arm6
        results_vis[std_frame_idx]['bodies']['candidate'][7,1] += y_offset_arm6
        results_vis[std_frame_idx]['hands'][0,:,0] += x_offset_arm6
        results_vis[std_frame_idx]['hands'][0,:,1] += y_offset_arm6

        ########arm7######## 6 --- 7 hands[0]
        l_arm7_ref = ((ref_candidate[7][0] - ref_candidate[6][0]) ** 2 + (ref_candidate[7][1] - ref_candidate[6][1]) ** 2) ** 0.5
        l_arm7_0 = ((candidate[7][0] - candidate[6][0]) ** 2 + (candidate[7][1] - candidate[6][1]) ** 2) ** 0.5

        arm7_ratio = l_arm7_ref / l_arm7_0

        x_offset_arm7 = (candidate[6][0]-candidate[7][0])*(1.-arm7_ratio)
        y_offset_arm7 = (candidate[6][1]-candidate[7][1])*(1.-arm7_ratio)

        align_parameters['ratio'].update({'arm7_ratio' : arm7_ratio})
        align_parameters['offset'].update({'offset_arm7' : [x_offset_arm7, y_offset_arm7]})

        results_vis[std_frame_idx]['bodies']['candidate'][7,0] += x_offset_arm7
        results_vis[std_frame_idx]['bodies']['candidate'][7,1] += y_offset_arm7
        results_vis[std_frame_idx]['hands'][0,:,0] += x_offset_arm7
        results_vis[std_frame_idx]['hands'][0,:,1] += y_offset_arm7

        ########head14######## 0 --- 14 16
        l_head14_ref = ((ref_candidate[14][0] - ref_candidate[0][0]) ** 2 + (ref_candidate[14][1] - ref_candidate[0][1]) ** 2) ** 0.5
        l_head14_0 = ((candidate[14][0] - candidate[0][0]) ** 2 + (candidate[14][1] - candidate[0][1]) ** 2) ** 0.5

        head14_ratio = l_head14_ref / l_head14_0

        x_offset_head14 = (candidate[0][0]-candidate[14][0])*(1.-head14_ratio)
        y_offset_head14 = (candidate[0][1]-candidate[14][1])*(1.-head14_ratio)

        align_parameters['ratio'].update({'head14_ratio' : head14_ratio})
        align_parameters['offset'].update({'offset_head14' : [x_offset_head14, y_offset_head14]})

        results_vis[std_frame_idx]['bodies']['candidate'][14,0] += x_offset_head14
        results_vis[std_frame_idx]['bodies']['candidate'][14,1] += y_offset_head14
        results_vis[std_frame_idx]['bodies']['candidate'][16,0] += x_offset_head14
        results_vis[std_frame_idx]['bodies']['candidate'][16,1] += y_offset_head14

        ########head15######## 0 --- 15 17
        l_head15_ref = ((ref_candidate[15][0] - ref_candidate[0][0]) ** 2 + (ref_candidate[15][1] - ref_candidate[0][1]) ** 2) ** 0.5
        l_head15_0 = ((candidate[15][0] - candidate[0][0]) ** 2 + (candidate[15][1] - candidate[0][1]) ** 2) ** 0.5

        head15_ratio = l_head15_ref / l_head15_0

        x_offset_head15 = (candidate[0][0]-candidate[15][0])*(1.-head15_ratio)
        y_offset_head15 = (candidate[0][1]-candidate[15][1])*(1.-head15_ratio)

        align_parameters['ratio'].update({'head15_ratio' : head15_ratio})
        align_parameters['offset'].update({'offset_head15' : [x_offset_head15, y_offset_head15]})

        results_vis[std_frame_idx]['bodies']['candidate'][15,0] += x_offset_head15
        results_vis[std_frame_idx]['bodies']['candidate'][15,1] += y_offset_head15
        results_vis[std_frame_idx]['bodies']['candidate'][17,0] += x_offset_head15
        results_vis[std_frame_idx]['bodies']['candidate'][17,1] += y_offset_head15

        ########head16######## 14 --- 16
        l_head16_ref = ((ref_candidate[16][0] - ref_candidate[14][0]) ** 2 + (ref_candidate[16][1] - ref_candidate[14][1]) ** 2) ** 0.5
        l_head16_0 = ((candidate[16][0] - candidate[14][0]) ** 2 + (candidate[16][1] - candidate[14][1]) ** 2) ** 0.5

        head16_ratio = l_head16_ref / l_head16_0

        x_offset_head16 = (candidate[14][0]-candidate[16][0])*(1.-head16_ratio)
        y_offset_head16 = (candidate[14][1]-candidate[16][1])*(1.-head16_ratio)

        align_parameters['ratio'].update({'head16_ratio' : head16_ratio})
        align_parameters['offset'].update({'offset_head16' : [x_offset_head16, y_offset_head16]})

        results_vis[std_frame_idx]['bodies']['candidate'][16,0] += x_offset_head16
        results_vis[std_frame_idx]['bodies']['candidate'][16,1] += y_offset_head16

        ########head17######## 15 --- 17
        l_head17_ref = ((ref_candidate[17][0] - ref_candidate[15][0]) ** 2 + (ref_candidate[17][1] - ref_candidate[15][1]) ** 2) ** 0.5
        l_head17_0 = ((candidate[17][0] - candidate[15][0]) ** 2 + (candidate[17][1] - candidate[15][1]) ** 2) ** 0.5

        head17_ratio = l_head17_ref / l_head17_0

        x_offset_head17 = (candidate[15][0]-candidate[17][0])*(1.-head17_ratio)
        y_offset_head17 = (candidate[15][1]-candidate[17][1])*(1.-head17_ratio)

        align_parameters['ratio'].update({'head17_ratio' : head17_ratio})
        align_parameters['offset'].update({'offset_head17' : [x_offset_head17, y_offset_head17]})

        results_vis[std_frame_idx]['bodies']['candidate'][17,0] += x_offset_head17
        results_vis[std_frame_idx]['bodies']['candidate'][17,1] += y_offset_head17
        
        ########MovingAverage########
        
        ########left leg######## 8 --- 9 10 19
        l_ll1_ref = ((ref_candidate[8][0] - ref_candidate[9][0]) ** 2 + (ref_candidate[8][1] - ref_candidate[9][1]) ** 2) ** 0.5
        l_ll1_0 = ((candidate[8][0] - candidate[9][0]) ** 2 + (candidate[8][1] - candidate[9][1]) ** 2) ** 0.5
        ll1_ratio = l_ll1_ref / l_ll1_0

        x_offset_ll1 = (candidate[9][0]-candidate[8][0])*(ll1_ratio-1.)
        y_offset_ll1 = (candidate[9][1]-candidate[8][1])*(ll1_ratio-1.)

        align_parameters['ratio'].update({'ll1_ratio' : ll1_ratio})
        align_parameters['offset'].update({'offset_ll1' : [x_offset_ll1, y_offset_ll1]})

        results_vis[std_frame_idx]['bodies']['candidate'][9,0] += x_offset_ll1
        results_vis[std_frame_idx]['bodies']['candidate'][9,1] += y_offset_ll1
        results_vis[std_frame_idx]['bodies']['candidate'][10,0] += x_offset_ll1
        results_vis[std_frame_idx]['bodies']['candidate'][10,1] += y_offset_ll1
        results_vis[std_frame_idx]['bodies']['candidate'][19,0] += x_offset_ll1
        results_vis[std_frame_idx]['bodies']['candidate'][19,1] += y_offset_ll1

        l_ll2_ref = ((ref_candidate[9][0] - ref_candidate[10][0]) ** 2 + (ref_candidate[9][1] - ref_candidate[10][1]) ** 2) ** 0.5
        l_ll2_0 = ((candidate[9][0] - candidate[10][0]) ** 2 + (candidate[9][1] - candidate[10][1]) ** 2) ** 0.5
        ll2_ratio = l_ll2_ref / l_ll2_0

        x_offset_ll2 = (candidate[10][0]-candidate[9][0])*(ll2_ratio-1.)
        y_offset_ll2 = (candidate[10][1]-candidate[9][1])*(ll2_ratio-1.)

        align_parameters['ratio'].update({'ll2_ratio' : ll2_ratio})
        align_parameters['offset'].update({'offset_ll2' : [x_offset_ll2, y_offset_ll2]})

        results_vis[std_frame_idx]['bodies']['candidate'][10,0] += x_offset_ll2
        results_vis[std_frame_idx]['bodies']['candidate'][10,1] += y_offset_ll2
        results_vis[std_frame_idx]['bodies']['candidate'][19,0] += x_offset_ll2
        results_vis[std_frame_idx]['bodies']['candidate'][19,1] += y_offset_ll2

        ########right leg######## 11 --- 12 13 18
        l_rl1_ref = ((ref_candidate[11][0] - ref_candidate[12][0]) ** 2 + (ref_candidate[11][1] - ref_candidate[12][1]) ** 2) ** 0.5
        l_rl1_0 = ((candidate[11][0] - candidate[12][0]) ** 2 + (candidate[11][1] - candidate[12][1]) ** 2) ** 0.5
        rl1_ratio = l_rl1_ref / l_rl1_0

        x_offset_rl1 = (candidate[12][0]-candidate[11][0])*(rl1_ratio-1.)
        y_offset_rl1 = (candidate[12][1]-candidate[11][1])*(rl1_ratio-1.)

        align_parameters['ratio'].update({'rl1_ratio' : rl1_ratio})
        align_parameters['offset'].update({'offset_rl1' : [x_offset_rl1, y_offset_rl1]})

        results_vis[std_frame_idx]['bodies']['candidate'][12,0] += x_offset_rl1
        results_vis[std_frame_idx]['bodies']['candidate'][12,1] += y_offset_rl1
        results_vis[std_frame_idx]['bodies']['candidate'][13,0] += x_offset_rl1
        results_vis[std_frame_idx]['bodies']['candidate'][13,1] += y_offset_rl1
        results_vis[std_frame_idx]['bodies']['candidate'][18,0] += x_offset_rl1
        results_vis[std_frame_idx]['bodies']['candidate'][18,1] += y_offset_rl1

        l_rl2_ref = ((ref_candidate[12][0] - ref_candidate[13][0]) ** 2 + (ref_candidate[12][1] - ref_candidate[13][1]) ** 2) ** 0.5
        l_rl2_0 = ((candidate[12][0] - candidate[13][0]) ** 2 + (candidate[12][1] - candidate[13][1]) ** 2) ** 0.5
        rl2_ratio = l_rl2_ref / l_rl2_0

        x_offset_rl2 = (candidate[13][0]-candidate[12][0])*(rl2_ratio-1.)
        y_offset_rl2 = (candidate[13][1]-candidate[12][1])*(rl2_ratio-1.)

        align_parameters['ratio'].update({'rl2_ratio' : rl2_ratio})
        align_parameters['offset'].update({'offset_rl2' : [x_offset_rl2, y_offset_rl2]})

        results_vis[std_frame_idx]['bodies']['candidate'][13,0] += x_offset_rl2
        results_vis[std_frame_idx]['bodies']['candidate'][13,1] += y_offset_rl2
        results_vis[std_frame_idx]['bodies']['candidate'][18,0] += x_offset_rl2
        results_vis[std_frame_idx]['bodies']['candidate'][18,1] += y_offset_rl2

        print(align_parameters)

        # breakpoint()
        # 统计各个部位的ratio clip极端数值
        ratio = list(align_parameters['ratio'].values())
        ratio = np.array(ratio)
        ratio_mean = np.mean(ratio)
        ratio_std = np.std(ratio)
        threshold = 0.5 * ratio_std
        lower_bound = ratio_mean - threshold  
        upper_bound = ratio_mean + threshold 
        for k, v in align_parameters['ratio'].items():
            if v < lower_bound:
                align_parameters['ratio'][k] = lower_bound
            elif v > upper_bound:
                align_parameters['ratio'][k] = upper_bound

        # new_ratio = np.array([r for r in ratio if r > lower_bound and r < upper_bound])
        # new_ratio_mean = new_ratio.sum()/len(new_ratio)
        
        # for k, v in align_parameters['ratio'].items():
        #     if v < lower_bound or v > upper_bound:
        #         align_parameters['ratio'][k] = new_ratio_mean
        
        # 左右ratio做平均处理
        shoulder2_5_ratio = (align_parameters['ratio']['shoulder2_ratio'] + align_parameters['ratio']['shoulder5_ratio']) / 2.0
        align_parameters['ratio']['shoulder2_ratio'], align_parameters['ratio']['shoulder5_ratio'] = shoulder2_5_ratio, shoulder2_5_ratio

        arm3_6_ratio = (align_parameters['ratio']['arm3_ratio'] + align_parameters['ratio']['arm6_ratio']) / 2.0
        align_parameters['ratio']['arm3_ratio'], align_parameters['ratio']['arm6_ratio'] = arm3_6_ratio, arm3_6_ratio

        arm4_7_ratio = (align_parameters['ratio']['arm4_ratio'] + align_parameters['ratio']['arm7_ratio']) / 2.0
        align_parameters['ratio']['arm4_ratio'], align_parameters['ratio']['arm7_ratio'] = arm4_7_ratio, arm4_7_ratio

        head14_15_ratio = (align_parameters['ratio']['head14_ratio'] + align_parameters['ratio']['head15_ratio']) / 2.0
        align_parameters['ratio']['head14_ratio'], align_parameters['ratio']['head15_ratio'] = head14_15_ratio, head14_15_ratio

        head16_17_ratio = (align_parameters['ratio']['head16_ratio'] + align_parameters['ratio']['head17_ratio']) / 2.0
        align_parameters['ratio']['head16_ratio'], align_parameters['ratio']['head17_ratio'] = head16_17_ratio, head16_17_ratio

        ll1_rl1_ratio = (align_parameters['ratio']['ll1_ratio'] + align_parameters['ratio']['rl1_ratio']) / 2.0
        align_parameters['ratio']['ll1_ratio'], align_parameters['ratio']['rl1_ratio'] = ll1_rl1_ratio, ll1_rl1_ratio

        ll2_rl2_ratio = (align_parameters['ratio']['ll2_ratio'] + align_parameters['ratio']['rl2_ratio']) / 2.0
        align_parameters['ratio']['ll2_ratio'], align_parameters['ratio']['rl2_ratio'] = ll2_rl2_ratio, ll2_rl2_ratio

        print(align_parameters)

        # ============================ stage 3 ============================
        # 全身以['bodies']['candidate'][1] 为参考 进行平移
        offset = ref_candidate[1] - results_vis[std_frame_idx]['bodies']['candidate'][1]

        results_vis[std_frame_idx]['bodies']['candidate'] += offset[np.newaxis, :]
        results_vis[std_frame_idx]['faces'] += offset[np.newaxis, np.newaxis, :]
        results_vis[std_frame_idx]['hands'] += offset[np.newaxis, np.newaxis, :]

        # ============================ stage 4 ============================
        neck_ratio = align_parameters['ratio']['neck_ratio']
        shoulder2_ratio = align_parameters['ratio']['shoulder2_ratio']
        shoulder5_ratio = align_parameters['ratio']['shoulder5_ratio']

        arm3_ratio = align_parameters['ratio']['arm3_ratio']
        arm6_ratio = align_parameters['ratio']['arm6_ratio']
        arm4_ratio = align_parameters['ratio']['arm4_ratio']
        arm7_ratio = align_parameters['ratio']['arm7_ratio']

        head14_ratio = align_parameters['ratio']['head14_ratio']
        head15_ratio = align_parameters['ratio']['head15_ratio']
        head16_ratio = align_parameters['ratio']['head16_ratio']
        head17_ratio = align_parameters['ratio']['head17_ratio']

        ll1_ratio = align_parameters['ratio']['ll1_ratio']
        ll2_ratio = align_parameters['ratio']['ll2_ratio']
        rl1_ratio = align_parameters['ratio']['rl1_ratio']
        rl2_ratio = align_parameters['ratio']['rl2_ratio']

        results_vis[std_frame_idx] = first_frame_pose # 修改第一帧

        for i in range(0, len(results_vis)):
            results_vis[i]['bodies']['candidate'][:,0] *= x_ratio
            results_vis[i]['bodies']['candidate'][:,1] *= y_ratio
            results_vis[i]['faces'][:,:,0] *= x_ratio
            results_vis[i]['faces'][:,:,1] *= y_ratio
            results_vis[i]['hands'][:,:,0] *= x_ratio
            results_vis[i]['hands'][:,:,1] *= y_ratio

            ########neck########
            x_offset_neck = (results_vis[i]['bodies']['candidate'][1][0]-results_vis[i]['bodies']['candidate'][0][0])*(1.-neck_ratio)
            y_offset_neck = (results_vis[i]['bodies']['candidate'][1][1]-results_vis[i]['bodies']['candidate'][0][1])*(1.-neck_ratio)

            results_vis[i]['bodies']['candidate'][0,0] += x_offset_neck
            results_vis[i]['bodies']['candidate'][0,1] += y_offset_neck
            results_vis[i]['bodies']['candidate'][14,0] += x_offset_neck
            results_vis[i]['bodies']['candidate'][14,1] += y_offset_neck
            results_vis[i]['bodies']['candidate'][15,0] += x_offset_neck
            results_vis[i]['bodies']['candidate'][15,1] += y_offset_neck
            results_vis[i]['bodies']['candidate'][16,0] += x_offset_neck
            results_vis[i]['bodies']['candidate'][16,1] += y_offset_neck
            results_vis[i]['bodies']['candidate'][17,0] += x_offset_neck
            results_vis[i]['bodies']['candidate'][17,1] += y_offset_neck

            ########shoulder2########
            

            x_offset_shoulder2 = (results_vis[i]['bodies']['candidate'][1][0]-results_vis[i]['bodies']['candidate'][2][0])*(1.-shoulder2_ratio)
            y_offset_shoulder2 = (results_vis[i]['bodies']['candidate'][1][1]-results_vis[i]['bodies']['candidate'][2][1])*(1.-shoulder2_ratio)

            results_vis[i]['bodies']['candidate'][2,0] += x_offset_shoulder2
            results_vis[i]['bodies']['candidate'][2,1] += y_offset_shoulder2
            results_vis[i]['bodies']['candidate'][3,0] += x_offset_shoulder2
            results_vis[i]['bodies']['candidate'][3,1] += y_offset_shoulder2
            results_vis[i]['bodies']['candidate'][4,0] += x_offset_shoulder2
            results_vis[i]['bodies']['candidate'][4,1] += y_offset_shoulder2
            results_vis[i]['hands'][1,:,0] += x_offset_shoulder2
            results_vis[i]['hands'][1,:,1] += y_offset_shoulder2

            ########shoulder5########

            x_offset_shoulder5 = (results_vis[i]['bodies']['candidate'][1][0]-results_vis[i]['bodies']['candidate'][5][0])*(1.-shoulder5_ratio)
            y_offset_shoulder5 = (results_vis[i]['bodies']['candidate'][1][1]-results_vis[i]['bodies']['candidate'][5][1])*(1.-shoulder5_ratio)

            results_vis[i]['bodies']['candidate'][5,0] += x_offset_shoulder5
            results_vis[i]['bodies']['candidate'][5,1] += y_offset_shoulder5
            results_vis[i]['bodies']['candidate'][6,0] += x_offset_shoulder5
            results_vis[i]['bodies']['candidate'][6,1] += y_offset_shoulder5
            results_vis[i]['bodies']['candidate'][7,0] += x_offset_shoulder5
            results_vis[i]['bodies']['candidate'][7,1] += y_offset_shoulder5
            results_vis[i]['hands'][0,:,0] += x_offset_shoulder5
            results_vis[i]['hands'][0,:,1] += y_offset_shoulder5

            ########arm3########

            x_offset_arm3 = (results_vis[i]['bodies']['candidate'][2][0]-results_vis[i]['bodies']['candidate'][3][0])*(1.-arm3_ratio)
            y_offset_arm3 = (results_vis[i]['bodies']['candidate'][2][1]-results_vis[i]['bodies']['candidate'][3][1])*(1.-arm3_ratio)

            results_vis[i]['bodies']['candidate'][3,0] += x_offset_arm3
            results_vis[i]['bodies']['candidate'][3,1] += y_offset_arm3
            results_vis[i]['bodies']['candidate'][4,0] += x_offset_arm3
            results_vis[i]['bodies']['candidate'][4,1] += y_offset_arm3
            results_vis[i]['hands'][1,:,0] += x_offset_arm3
            results_vis[i]['hands'][1,:,1] += y_offset_arm3

            ########arm4########

            x_offset_arm4 = (results_vis[i]['bodies']['candidate'][3][0]-results_vis[i]['bodies']['candidate'][4][0])*(1.-arm4_ratio)
            y_offset_arm4 = (results_vis[i]['bodies']['candidate'][3][1]-results_vis[i]['bodies']['candidate'][4][1])*(1.-arm4_ratio)

            results_vis[i]['bodies']['candidate'][4,0] += x_offset_arm4
            results_vis[i]['bodies']['candidate'][4,1] += y_offset_arm4
            results_vis[i]['hands'][1,:,0] += x_offset_arm4
            results_vis[i]['hands'][1,:,1] += y_offset_arm4

            ########arm6########

            x_offset_arm6 = (results_vis[i]['bodies']['candidate'][5][0]-results_vis[i]['bodies']['candidate'][6][0])*(1.-arm6_ratio)
            y_offset_arm6 = (results_vis[i]['bodies']['candidate'][5][1]-results_vis[i]['bodies']['candidate'][6][1])*(1.-arm6_ratio)

            results_vis[i]['bodies']['candidate'][6,0] += x_offset_arm6
            results_vis[i]['bodies']['candidate'][6,1] += y_offset_arm6
            results_vis[i]['bodies']['candidate'][7,0] += x_offset_arm6
            results_vis[i]['bodies']['candidate'][7,1] += y_offset_arm6
            results_vis[i]['hands'][0,:,0] += x_offset_arm6
            results_vis[i]['hands'][0,:,1] += y_offset_arm6

            ########arm7########

            x_offset_arm7 = (results_vis[i]['bodies']['candidate'][6][0]-results_vis[i]['bodies']['candidate'][7][0])*(1.-arm7_ratio)
            y_offset_arm7 = (results_vis[i]['bodies']['candidate'][6][1]-results_vis[i]['bodies']['candidate'][7][1])*(1.-arm7_ratio)

            results_vis[i]['bodies']['candidate'][7,0] += x_offset_arm7
            results_vis[i]['bodies']['candidate'][7,1] += y_offset_arm7
            results_vis[i]['hands'][0,:,0] += x_offset_arm7
            results_vis[i]['hands'][0,:,1] += y_offset_arm7

            ########head14########

            x_offset_head14 = (results_vis[i]['bodies']['candidate'][0][0]-results_vis[i]['bodies']['candidate'][14][0])*(1.-head14_ratio)
            y_offset_head14 = (results_vis[i]['bodies']['candidate'][0][1]-results_vis[i]['bodies']['candidate'][14][1])*(1.-head14_ratio)

            results_vis[i]['bodies']['candidate'][14,0] += x_offset_head14
            results_vis[i]['bodies']['candidate'][14,1] += y_offset_head14
            results_vis[i]['bodies']['candidate'][16,0] += x_offset_head14
            results_vis[i]['bodies']['candidate'][16,1] += y_offset_head14

            ########head15########

            x_offset_head15 = (results_vis[i]['bodies']['candidate'][0][0]-results_vis[i]['bodies']['candidate'][15][0])*(1.-head15_ratio)
            y_offset_head15 = (results_vis[i]['bodies']['candidate'][0][1]-results_vis[i]['bodies']['candidate'][15][1])*(1.-head15_ratio)

            results_vis[i]['bodies']['candidate'][15,0] += x_offset_head15
            results_vis[i]['bodies']['candidate'][15,1] += y_offset_head15
            results_vis[i]['bodies']['candidate'][17,0] += x_offset_head15
            results_vis[i]['bodies']['candidate'][17,1] += y_offset_head15

            ########head16########

            x_offset_head16 = (results_vis[i]['bodies']['candidate'][14][0]-results_vis[i]['bodies']['candidate'][16][0])*(1.-head16_ratio)
            y_offset_head16 = (results_vis[i]['bodies']['candidate'][14][1]-results_vis[i]['bodies']['candidate'][16][1])*(1.-head16_ratio)

            results_vis[i]['bodies']['candidate'][16,0] += x_offset_head16
            results_vis[i]['bodies']['candidate'][16,1] += y_offset_head16

            ########head17########
            x_offset_head17 = (results_vis[i]['bodies']['candidate'][15][0]-results_vis[i]['bodies']['candidate'][17][0])*(1.-head17_ratio)
            y_offset_head17 = (results_vis[i]['bodies']['candidate'][15][1]-results_vis[i]['bodies']['candidate'][17][1])*(1.-head17_ratio)

            results_vis[i]['bodies']['candidate'][17,0] += x_offset_head17
            results_vis[i]['bodies']['candidate'][17,1] += y_offset_head17

            # ########MovingAverage########

            ########left leg########
            x_offset_ll1 = (results_vis[i]['bodies']['candidate'][9][0]-results_vis[i]['bodies']['candidate'][8][0])*(ll1_ratio-1.)
            y_offset_ll1 = (results_vis[i]['bodies']['candidate'][9][1]-results_vis[i]['bodies']['candidate'][8][1])*(ll1_ratio-1.)

            results_vis[i]['bodies']['candidate'][9,0] += x_offset_ll1
            results_vis[i]['bodies']['candidate'][9,1] += y_offset_ll1
            results_vis[i]['bodies']['candidate'][10,0] += x_offset_ll1
            results_vis[i]['bodies']['candidate'][10,1] += y_offset_ll1
            results_vis[i]['bodies']['candidate'][19,0] += x_offset_ll1
            results_vis[i]['bodies']['candidate'][19,1] += y_offset_ll1



            x_offset_ll2 = (results_vis[i]['bodies']['candidate'][10][0]-results_vis[i]['bodies']['candidate'][9][0])*(ll2_ratio-1.)
            y_offset_ll2 = (results_vis[i]['bodies']['candidate'][10][1]-results_vis[i]['bodies']['candidate'][9][1])*(ll2_ratio-1.)

            results_vis[i]['bodies']['candidate'][10,0] += x_offset_ll2
            results_vis[i]['bodies']['candidate'][10,1] += y_offset_ll2
            results_vis[i]['bodies']['candidate'][19,0] += x_offset_ll2
            results_vis[i]['bodies']['candidate'][19,1] += y_offset_ll2

            ########right leg########

            x_offset_rl1 = (results_vis[i]['bodies']['candidate'][12][0]-results_vis[i]['bodies']['candidate'][11][0])*(rl1_ratio-1.)
            y_offset_rl1 = (results_vis[i]['bodies']['candidate'][12][1]-results_vis[i]['bodies']['candidate'][11][1])*(rl1_ratio-1.)

            results_vis[i]['bodies']['candidate'][12,0] += x_offset_rl1
            results_vis[i]['bodies']['candidate'][12,1] += y_offset_rl1
            results_vis[i]['bodies']['candidate'][13,0] += x_offset_rl1
            results_vis[i]['bodies']['candidate'][13,1] += y_offset_rl1
            results_vis[i]['bodies']['candidate'][18,0] += x_offset_rl1
            results_vis[i]['bodies']['candidate'][18,1] += y_offset_rl1


            x_offset_rl2 = (results_vis[i]['bodies']['candidate'][13][0]-results_vis[i]['bodies']['candidate'][12][0])*(rl2_ratio-1.)
            y_offset_rl2 = (results_vis[i]['bodies']['candidate'][13][1]-results_vis[i]['bodies']['candidate'][12][1])*(rl2_ratio-1.)

            results_vis[i]['bodies']['candidate'][13,0] += x_offset_rl2
            results_vis[i]['bodies']['candidate'][13,1] += y_offset_rl2
            results_vis[i]['bodies']['candidate'][18,0] += x_offset_rl2
            results_vis[i]['bodies']['candidate'][18,1] += y_offset_rl2

            results_vis[i]['bodies']['candidate'] += offset[np.newaxis, :]
            results_vis[i]['faces'] += offset[np.newaxis, np.newaxis, :]
            results_vis[i]['hands'] += offset[np.newaxis, np.newaxis, :]
        
        for i in range(len(results_vis)):
            dwpose_woface, dwpose_wface = draw_pose(results_vis[i], H=1216*3, W=768*3)
            img_path = save_motion+'/' + str(i).zfill(4) + '.jpg'
            cv2.imwrite(img_path, dwpose_woface)
        
        dwpose_woface, dwpose_wface = draw_pose(pose_ref, H=1216*3, W=768*3)
        img_path = save_warp+'/' + 'ref_pose.jpg'
        cv2.imwrite(img_path, dwpose_woface)

        print('='*50)
        print(align_parameters['ratio'])


logger = get_logger('dw pose extraction')


if __name__=='__main__':
    def parse_args(): 
        parser = argparse.ArgumentParser(description="Simple example of a training script.")
        parser.add_argument("--ref_name", type=str, default="data/images/IMG_20240514_104337.jpg",)
        parser.add_argument("--source_video_paths", type=str, default="data/videos/source_video.mp4",)
        parser.add_argument("--saved_pose_dir", type=str, default="data/saved_pose/IMG_20240514_104337",)
        parser.add_argument("--std_frame_idx", type=int, default=-1,)
        args = parser.parse_args()

        return args
        
    args = parse_args()
    mp_main(args)
    

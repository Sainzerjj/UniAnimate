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
import zipfile
# import skvideo.io
import numpy as np
from io import BytesIO
import urllib.request
import torch.nn.functional as F
import torchvision.utils as tvutils
from multiprocessing.pool import ThreadPool as Pool
from einops import rearrange
from PIL import Image, ImageDraw, ImageFont
from IPython import embed


def gen_text_image(captions, text_size):
    num_char = int(38 * (text_size / text_size))
    font_size = int(text_size / 20)
    font = ImageFont.truetype('data/font/DejaVuSans.ttf', size=font_size)
    text_image_list = []
    for text in captions:
        txt_img = Image.new("RGB", (text_size, text_size), color="white") 
        draw = ImageDraw.Draw(txt_img)
        lines = "\n".join(text[start:start + num_char] for start in range(0, len(text), num_char))
        draw.text((0, 0), lines, fill="black", font=font)
        txt_img = np.array(txt_img)
        text_image_list.append(txt_img)
    text_images = np.stack(text_image_list, axis=0)
    text_images = torch.from_numpy(text_images)
    return text_images

@torch.no_grad()
def save_video_refimg_and_text(
    local_path,
    ref_frame,
    gen_video, 
    captions, 
    mean=[0.5, 0.5, 0.5], 
    std=[0.5, 0.5, 0.5], 
    text_size=256, 
    nrow=4, 
    save_fps=8,
    retry=5):
    ''' 
    gen_video: BxCxFxHxW
    '''
    nrow = max(int(gen_video.size(0) / 2), 1)
    vid_mean = torch.tensor(mean, device=gen_video.device).view(1, -1, 1, 1, 1) #ncfhw
    vid_std = torch.tensor(std, device=gen_video.device).view(1, -1, 1, 1, 1) #ncfhw

    text_images = gen_text_image(captions, text_size) # Tensor 8x256x256x3
    text_images = text_images.unsqueeze(1) # Tensor 8x1x256x256x3
    text_images = text_images.repeat_interleave(repeats=gen_video.size(2), dim=1) # 8x16x256x256x3

    ref_frame = ref_frame.unsqueeze(2)
    ref_frame = ref_frame.mul_(vid_std).add_(vid_mean)
    ref_frame = ref_frame.repeat_interleave(repeats=gen_video.size(2), dim=2) # 8x16x256x256x3
    ref_frame.clamp_(0, 1)
    ref_frame = ref_frame * 255.0
    ref_frame = rearrange(ref_frame, 'b c f h w -> b f h w c')
    
    gen_video = gen_video.mul_(vid_std).add_(vid_mean)  # 8x3x16x256x384
    gen_video.clamp_(0, 1)
    gen_video = gen_video * 255.0

    images = rearrange(gen_video, 'b c f h w -> b f h w c')
    images = torch.cat([ref_frame, images, text_images], dim=3)

    images = rearrange(images, '(r j) f h w c -> f (r h) (j w) c', r=nrow)
    images = [(img.numpy()).astype('uint8') for img in images]

    for _ in [None] * retry:
        try:
            if len(images) == 1:
                local_path = local_path + '.png'
                cv2.imwrite(local_path, images[0][:,:,::-1], [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            else:
                local_path = local_path + '.mp4'
                frame_dir = os.path.join(os.path.dirname(local_path), '%s_frames' % (os.path.basename(local_path)))
                os.system(f'rm -rf {frame_dir}'); os.makedirs(frame_dir, exist_ok=True)
                for fid, frame in enumerate(images):
                    tpth = os.path.join(frame_dir, '%04d.png' % (fid+1))
                    cv2.imwrite(tpth, frame[:,:,::-1], [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                cmd = f'ffmpeg -y -f image2 -loglevel quiet -framerate {save_fps} -i {frame_dir}/%04d.png -vcodec libx264 -crf 17  -pix_fmt yuv420p {local_path}'
                os.system(cmd); os.system(f'rm -rf {frame_dir}')
                # os.system(f'rm -rf {local_path}')
            exception = None
            break
        except Exception as e:
            exception = e
            continue


@torch.no_grad()
def save_i2vgen_video(
    local_path,
    image_id,
    gen_video, 
    captions, 
    mean=[0.5, 0.5, 0.5], 
    std=[0.5, 0.5, 0.5], 
    text_size=256, 
    retry=5,
    save_fps = 8
):
    ''' 
    Save both the generated video and the input conditions.
    '''
    vid_mean = torch.tensor(mean, device=gen_video.device).view(1, -1, 1, 1, 1) #ncfhw
    vid_std = torch.tensor(std, device=gen_video.device).view(1, -1, 1, 1, 1) #ncfhw

    text_images = gen_text_image(captions, text_size) # Tensor 1x256x256x3
    text_images = text_images.unsqueeze(1) # Tensor 1x1x256x256x3
    text_images = text_images.repeat_interleave(repeats=gen_video.size(2), dim=1) # 1x16x256x256x3

    image_id = image_id.unsqueeze(2) # B, C, F, H, W
    image_id = image_id.repeat_interleave(repeats=gen_video.size(2), dim=2) # 1x3x32x256x448
    image_id = image_id.mul_(vid_std).add_(vid_mean)  # 32x3x256x448
    image_id.clamp_(0, 1)
    image_id = image_id * 255.0
    image_id = rearrange(image_id, 'b c f h w -> b f h w c')

    gen_video = gen_video.mul_(vid_std).add_(vid_mean)  # 8x3x16x256x384
    gen_video.clamp_(0, 1)
    gen_video = gen_video * 255.0

    images = rearrange(gen_video, 'b c f h w -> b f h w c')
    images = torch.cat([image_id, images, text_images], dim=3)
    images = images[0]
    images = [(img.numpy()).astype('uint8') for img in images]

    exception = None
    for _ in [None] * retry:
        try:
            frame_dir = os.path.join(os.path.dirname(local_path), '%s_frames' % (os.path.basename(local_path)))
            os.system(f'rm -rf {frame_dir}'); os.makedirs(frame_dir, exist_ok=True)
            for fid, frame in enumerate(images):
                tpth = os.path.join(frame_dir, '%04d.png' % (fid+1))
                cv2.imwrite(tpth, frame[:,:,::-1], [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            cmd = f'ffmpeg -y -f image2 -loglevel quiet -framerate {save_fps} -i {frame_dir}/%04d.png -vcodec libx264 -crf 17  -pix_fmt yuv420p {local_path}'
            os.system(cmd); os.system(f'rm -rf {frame_dir}')
            break
        except Exception as e:
            exception = e
            continue
    
    if exception is not None:
        raise exception


@torch.no_grad()
def save_i2vgen_video_safe(
    local_path,
    gen_video, 
    captions, 
    ref_image_key, 
    mean=[0.5, 0.5, 0.5], 
    std=[0.5, 0.5, 0.5], 
    text_size=256, 
    retry=5,
    save_fps = 8
):
    '''
    Save only the generated video, do not save the related reference conditions, and at the same time perform anomaly detection on the last frame.
    '''
    vid_mean = torch.tensor(mean, device=gen_video.device).view(1, -1, 1, 1, 1) #ncfhw
    vid_std = torch.tensor(std, device=gen_video.device).view(1, -1, 1, 1, 1) #ncfhw

    gen_video = gen_video.mul_(vid_std).add_(vid_mean)  # 8x3x16x256x384
    gen_video.clamp_(0, 1)
    gen_video = gen_video * 255.0

    images = rearrange(gen_video, 'b c f h w -> b f h w c')
    images = images[0]
    images = [(img.numpy()).astype('uint8') for img in images]
    num_image = len(images)
    exception = None
    for _ in [None] * retry:
        try:
            if num_image == 1:
                local_path = local_path + '.png'
                cv2.imwrite(local_path, images[0][:,:,::-1], [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            else:
                writer = imageio.get_writer(local_path, fps=save_fps, codec='libx264', quality=8)
                for fid, frame in enumerate(images):
                    if fid == num_image-1: # Fix known bugs.
                        ratio = (np.sum((frame >= 117) & (frame <= 137)))/(frame.size)
                        if ratio > 0.4: continue
                    writer.append_data(frame)
                writer.close()
                # cmd = f'python ./facefusion/run.py -s {ref_image_key} -t {local_path} -o {local_path} --headless --execution-providers cuda --face-selector-mode one'
                # os.system(cmd)
            break
        except Exception as e:
            exception = e
            continue
    
    if exception is not None:
        raise exception


@torch.no_grad()
def save_t2vhigen_video_safe(
    local_path,
    gen_video, 
    captions, 
    mean=[0.5, 0.5, 0.5], 
    std=[0.5, 0.5, 0.5], 
    text_size=256, 
    retry=5,
    save_fps = 8
):
    '''
    Save only the generated video, do not save the related reference conditions, and at the same time perform anomaly detection on the last frame.
    '''
    vid_mean = torch.tensor(mean, device=gen_video.device).view(1, -1, 1, 1, 1) #ncfhw
    vid_std = torch.tensor(std, device=gen_video.device).view(1, -1, 1, 1, 1) #ncfhw

    gen_video = gen_video.mul_(vid_std).add_(vid_mean)  # 8x3x16x256x384
    gen_video.clamp_(0, 1)
    gen_video = gen_video * 255.0

    images = rearrange(gen_video, 'b c f h w -> b f h w c')
    images = images[0]
    images = [(img.numpy()).astype('uint8') for img in images]
    num_image = len(images)
    exception = None
    for _ in [None] * retry:
        try:
            if num_image == 1:
                local_path = local_path + '.png'
                cv2.imwrite(local_path, images[0][:,:,::-1], [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            else:
                with imageio.get_writer(local_path, fps=save_fps) as writer:
                    for img in images:   # c f h w
                        img_array = np.array(img)  # Convert PIL Image to numpy array
                        writer.append_data(img_array)
                # frame_dir = os.path.join(os.path.dirname(local_path), '%s_frames' % (os.path.basename(local_path)))
                # os.system(f'rm -rf {frame_dir}'); os.makedirs(frame_dir, exist_ok=True)
                # for fid, frame in enumerate(images):
                #     if fid == num_image-1: # Fix known bugs.
                #         ratio = (np.sum((frame >= 117) & (frame <= 137)))/(frame.size)
                #         if ratio > 0.4: continue
                #     tpth = os.path.join(frame_dir, '%04d.png' % (fid+1))
                #     cv2.imwrite(tpth, frame[:,:,::-1], [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                # cmd = f'ffmpeg -y -f image2 -framerate {save_fps} -i {frame_dir}/%04d.png -vcodec libx264 -crf 17  -pix_fmt yuv420p {local_path}'
                # os.system(cmd) 
                # os.system(f'rm -rf {frame_dir}')
            break
        except Exception as e:
            exception = e
            continue
    
    if exception is not None:
        raise exception




@torch.no_grad()
def save_video_multiple_conditions_not_gif_horizontal_3col(local_path, video_tensor, model_kwargs, source_imgs, 
                                   mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5], nrow=8, retry=5, save_fps=8):
    mean=torch.tensor(mean,device=video_tensor.device).view(1,-1,1,1,1)#ncfhw
    std=torch.tensor(std,device=video_tensor.device).view(1,-1,1,1,1)#ncfhw
    video_tensor = video_tensor.mul_(std).add_(mean)  #### unnormalize back to [0,1]
    video_tensor.clamp_(0, 1)

    b, c, n, h, w = video_tensor.shape  # [1, 3, 32, 1216, 768]
    source_imgs = F.adaptive_avg_pool3d(source_imgs, (n, h, w))
    source_imgs = source_imgs.cpu()

    model_kwargs_channel3 = {}
    for key, conditions in model_kwargs[0].items():
        if key != 'ori_video':
            if conditions.size(1) == 1:
                conditions = torch.cat([conditions, conditions, conditions], dim=1)
                conditions = F.adaptive_avg_pool3d(conditions, (n, h, w))
            if conditions.size(1) == 2:
                conditions = torch.cat([conditions, conditions[:,:1,]], dim=1)
                conditions = F.adaptive_avg_pool3d(conditions, (n, h, w))
            elif conditions.size(1) == 3:
                conditions = F.adaptive_avg_pool3d(conditions, (n, h, w))
            elif conditions.size(1) == 4: # means it is a mask.
                color = ((conditions[:, 0:3] + 1.)/2.) # .astype(np.float32)
                alpha = conditions[:, 3:4] # .astype(np.float32)
                conditions = color * alpha + 1.0 * (1.0 - alpha)
                conditions = F.adaptive_avg_pool3d(conditions, (n, h, w))
        model_kwargs_channel3[key] = conditions.cpu() if conditions.is_cuda else conditions

    # filename = rand_name(suffix='.gif')
    for _ in [None] * retry:
        try:
            vid_gif = rearrange(video_tensor, '(i j) c f h w -> c f (i h) (j w)', i = nrow)
            
            cons_list = [rearrange(con, '(i j) c f h w -> c f (i h) (j w)', i = nrow) for _, con in model_kwargs_channel3.items()]
            # vid_gif = torch.cat(cons_list + [vid_gif,], dim=3)
            
            vid_gif = vid_gif.permute(1,2,3,0)  # f h w c
            
            images = vid_gif * 255.0
            images = [(img.numpy()).astype('uint8') for img in images]
            if len(images) == 1:
                
                local_path = local_path.replace('.mp4', '.png')
                cv2.imwrite(local_path, images[0][:,:,::-1], [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                # bucket.put_object_from_file(oss_key, local_path)
            else:

                outputs = []
                for image_name in images:
                    x = Image.fromarray(image_name)  # h, w, c
                    outputs.append(x)
                from pathlib import Path
                save_fmt = Path(local_path).suffix
                
                if save_fmt == ".mp4":
                    with imageio.get_writer(local_path, fps=save_fps) as writer:
                        for img in outputs:   # f h w c
                            img_array = np.array(img)  # Convert PIL Image to numpy array
                            writer.append_data(img_array)

                elif save_fmt == ".gif":
                    outputs[0].save(
                        fp=local_path,
                        format="GIF",
                        append_images=outputs[1:],
                        save_all=True,
                        duration=(1 / save_fps * 1000),
                        loop=0,
                    )
                else:
                    raise ValueError("Unsupported file type. Use .mp4 or .gif.")

                # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                # fps = save_fps
                # image = images[0] 
                # media_writer = cv2.VideoWriter(local_path, fourcc, fps, (image.shape[1],image.shape[0]))
                # for image_name in images:
                #     im = image_name[:,:,::-1] 
                #     media_writer.write(im)
                # media_writer.release()
                
            
            exception = None
            break
        except Exception as e:
            exception = e
            continue
    if exception is not None:
        print('save video to {} failed, error: {}'.format(local_path, exception), flush=True)
    

# ffmpeg -y -f image2 -framerate 8 -i outputs/UniAnimate_infer/rank_01_00_00_seed_11_image_local_image_dwpose_1_1216x768.mp4_frames/%04d.png -vcodec libx264 -crf 17  -pix_fmt yuv420p outputs/UniAnimate_infer/test

def save_tensor_as_video(tensor, output_file, fps=8):
    """
    将范围在 [0, 1] 之间的 PyTorch 张量保存为视频文件
    参数:
        tensor (torch.Tensor): 形状为 (1, 3, T, H, W) 的张量，其中:
            - 1 是批量大小（假设为1）
            - 3 是通道数（RGB）
            - T 是帧数
            - H 和 W 分别是高度和宽度
        output_file (str): 输出视频文件的路径
        fps (int): 视频帧率
    返回:
        None
    """
    # 确保输入张量的形状为 (1, 3, T, H, W)
    assert tensor.ndimension() == 5 and tensor.shape[0] == 1 and tensor.shape[1] == 3, \
        "张量的形状应该为 (1, 3, T, H, W)"
    # 去掉批次维度，得到形状为 (3, T, H, W)
    tensor = tensor.squeeze(0)
    # 重新调整值范围从 [0, 1] 到 [0, 255]
    tensor = (tensor * 255).byte()
    # 转换为 NumPy 数组，形状为 (3, T, H, W)
    tensor_np = tensor.numpy()
    # 转换为形状 (T, H, W, 3)，以便 OpenCV 处理
    tensor_np = tensor_np.transpose((1, 2, 3, 0))
    # 获取宽度和高度
    h, w = tensor_np.shape[1:3]
    with imageio.get_writer(output_file, fps=fps) as writer:
        for img in tensor_np:   # c f h w
            img_array = np.array(img)  # Convert PIL Image to numpy array
            writer.append_data(img_array)
    print(f"视频已保存到 {output_file}")


def read_video_to_tensor(video_path, transform=None):
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频文件：{video_path}")
        return None
    frames = []
    # 读取视频帧
    while True:
        ret, frame = cap.read()
        if not ret:
            break    
        # OpenCV 默认读取的是 BGR 格式，将其转换为 RGB 格式
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 转换为 PIL 图像
        frame = Image.fromarray(frame) 
        # 如果有定义的变换，则应用
        if transform:
            frame = transform(frame) 
        else:
            import utils.transforms as data
            frame = data.ToTensor()(frame)
        # 将处理后的帧追加到 frames 列表中
        frames.append(frame)
    cap.release()
    # 将帧列表转换为张量；形状为 (num_frames, channels, height, width)
    frames_tensor = torch.stack(frames)
    frames_tensor = frames_tensor.permute(1, 0, 2, 3).unsqueeze(0)
    return frames_tensor
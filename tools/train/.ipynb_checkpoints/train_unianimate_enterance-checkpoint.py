import os
import os.path as osp
import sys
sys.path.insert(0, '/'.join(osp.realpath(__file__).split('/')[:-4]))
import json
import math
import random
import torch
import logging
import pynvml
import datetime
import numpy as np
from PIL import Image
import torch.optim as optim 
from einops import rearrange
import torch.cuda.amp as amp
from importlib import reload
from copy import deepcopy, copy
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
import cv2

import utils.transforms as data
from utils.util import to_device
from ..modules.config import cfg
from utils.seed import setup_seed
from utils.optim import AnnealingLR
from utils.multi_port import find_free_port
from utils.assign_cfg import assign_signle_cfg
from utils.distributed import generalized_all_gather, all_reduce
from utils.video_op import save_i2vgen_video, save_t2vhigen_video_safe, save_video_multiple_conditions_not_gif_horizontal_3col
from tools.modules.autoencoder import get_first_stage_encoding
from utils.registry_class import ENGINE, MODEL, DATASETS, EMBEDDER, EMBEDMANAGER, AUTO_ENCODER, DISTRIBUTION, VISUAL, DIFFUSION, PRETRAIN


@ENGINE.register_function()
def train_unianimate_entrance(cfg_update,  **kwargs):
    for k, v in cfg_update.items():
        if isinstance(v, dict) and k in cfg:
            cfg[k].update(v)
        else:
            cfg[k] = v
    
    if not 'MASTER_ADDR' in os.environ:
        os.environ['MASTER_ADDR']='localhost'
        os.environ['MASTER_PORT']= find_free_port()
    cfg.pmi_rank = int(os.getenv('RANK', 0)) # 0
    cfg.pmi_world_size = int(os.getenv('WORLD_SIZE', 1))

    cfg.debug = True
    if cfg.debug:
        cfg.gpus_per_machine = 1
        cfg.world_size = 1
    else:
        cfg.gpus_per_machine = torch.cuda.device_count()
        cfg.world_size = cfg.pmi_world_size * cfg.gpus_per_machine
    
    if cfg.world_size == 1:
        worker(0, cfg, cfg_update)
    else:
        mp.spawn(worker, nprocs=cfg.gpus_per_machine, args=(cfg, cfg_update))
    return cfg

def make_masked_images(imgs, masks):
    masked_imgs = []
    for i, mask in enumerate(masks):        
        # concatenation
        masked_imgs.append(torch.cat([imgs[i] * (1 - mask), (1 - mask)], dim=1))
    return torch.stack(masked_imgs, dim=0)

def worker(gpu, cfg, cfg_update):
    '''
    Training worker for each gpu
    '''
    cfg.gpu = gpu
    cfg.seed = int(cfg.seed)
    cfg.rank = cfg.pmi_rank * cfg.gpus_per_machine + gpu
    setup_seed(cfg.seed + cfg.rank)
    
    if not cfg.debug:
        torch.cuda.set_device(gpu)
        torch.backends.cudnn.benchmark = True
        dist.init_process_group(backend='nccl', world_size=cfg.world_size, rank=cfg.rank)
    
    # [Log] Save logging
    log_dir = generalized_all_gather(cfg.log_dir)[0]
    inf_name = osp.basename(cfg.cfg_file).split('.')[0]

    cfg.log_dir = osp.join(cfg.log_dir, inf_name)
    os.makedirs(cfg.log_dir, exist_ok=True)
    if cfg.rank == 0:
        log_file = osp.join(cfg.log_dir, 'log.txt')
        cfg.log_file = log_file
        reload(logging)
        logging.basicConfig(
            level=logging.INFO,
            format='[%(asctime)s] %(levelname)s: %(message)s',
            handlers=[
                logging.FileHandler(filename=log_file),
                logging.StreamHandler(stream=sys.stdout)])
        logging.info(cfg)
        logging.info(f'Save all the file in to dir {cfg.log_dir}')
        logging.info(f"Going into i2v_img_fullid_vidcom function on {gpu} gpu")

    # [Diffusion]  build diffusion settings
    diffusion = DIFFUSION.build(cfg.Diffusion)

    # [Dataset] imagedataset and videodataset
    len_frames = len(cfg.frame_lens) # 8  frame_lens: [16, 16, 16, 16, 16, 32, 32, 32]
    len_fps = len(cfg.sample_fps) # 8  sample_fps: [8,  8,  16, 16, 16, 8,  16, 16]
    cfg.max_frames = cfg.frame_lens[cfg.rank % len_frames]
    cfg.batch_size = cfg.batch_sizes[str(cfg.max_frames)] # 1 2
    cfg.sample_fps = cfg.sample_fps[cfg.rank % len_fps]
    
    if cfg.rank == 0:
        logging.info(f'Currnt worker with max_frames={cfg.max_frames}, batch_size={cfg.batch_size}, sample_fps={cfg.sample_fps}')

    # [Data] Data Transform    
    train_trans = data.Compose([
        data.Resize(cfg.resolution),
        data.ToTensor(),
        data.Normalize(mean=cfg.mean, std=cfg.std)
        ])

    train_trans_pose = data.Compose([
        data.Resize(cfg.resolution),
        data.ToTensor(),
        ]
    )

    train_trans_vit = T.Compose([
                data.Resize(cfg.vit_resolution),
                T.ToTensor(),
                T.Normalize(mean=cfg.vit_mean, std=cfg.vit_std)])
    
    if cfg.max_frames == 1:
        cfg.sample_fps = 1
        dataset = DATASETS.build(cfg.img_dataset, transforms=train_trans, vit_transforms=train_trans_vit)
    else:
        dataset = DATASETS.build(cfg.vid_dataset, sample_fps=cfg.sample_fps, transforms=train_trans, vit_transforms=train_trans_vit, pose_transforms=train_trans_pose, max_frames=cfg.max_frames)
    
    sampler = DistributedSampler(dataset, num_replicas=cfg.world_size, rank=cfg.rank) if (cfg.world_size > 1 and not cfg.debug) else None
    dataloader = DataLoader(
        dataset, 
        sampler=sampler,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True,
        prefetch_factor=cfg.prefetch_factor)
    rank_iter = iter(dataloader) 
    
    # [Model] embedder
    clip_encoder = EMBEDDER.build(cfg.embedder)
    clip_encoder.model.to(gpu)
    with torch.no_grad():
        _, _, zero_y = clip_encoder(text="")

    # [Model] auotoencoder 
    autoencoder = AUTO_ENCODER.build(cfg.auto_encoder)
    autoencoder.eval() # freeze
    for param in autoencoder.parameters():
        param.requires_grad = False
    autoencoder.cuda()

    # [Model] UNet 
    if "config" in cfg.UNet:
        cfg.UNet["config"] = cfg
    cfg.UNet["zero_y"] = zero_y
    model = MODEL.build(cfg.UNet)
    resume_step = 0

    if cfg.checkpoint_model:
        state_dict = torch.load(cfg.checkpoint_model, map_location='cpu')
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        if 'step' in state_dict:
            resume_step = state_dict['step']
        status = model.load_state_dict(state_dict, strict=True)
        logging.info('Load model from {} with status {}'.format(cfg.checkpoint_model, status))
        
    model = model.to(gpu)

    if cfg.use_fsdp:
        config = {}
        config['compute_dtype'] = torch.float32
        config['mixed_precision'] = True
        model = FSDP(model, **config)
    else:
        model = DistributedDataParallel(model, device_ids=[gpu]) if not cfg.debug else model.to(gpu)

    torch.cuda.empty_cache()

    if cfg.use_ema:
        ema = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
        ema = type(ema)([(k, ema[k].data.clone()) for k in list(ema.keys())[cfg.rank::cfg.world_size]])
    
    # optimizer
    optimizer = optim.AdamW(params=model.parameters(),
                        lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = amp.GradScaler(enabled=cfg.use_fp16)

    # scheduler
    scheduler = AnnealingLR(
        optimizer=optimizer,
        base_lr=cfg.lr,
        warmup_steps=cfg.warmup_steps,  # 10
        total_steps=cfg.num_steps,      # 200000
        decay_mode=cfg.decay_mode)      # 'cosine'
    
    for step in range(resume_step, cfg.num_steps + 1): 
        model.train()
        
        try:
            batch = next(rank_iter)
        except StopIteration:
            rank_iter = iter(dataloader)
            batch = next(rank_iter)

        batch = to_device(batch, gpu, non_blocking=True)
        
        vit_frame, video_data, misc_data, dwpose_data, random_ref_frame_data, random_ref_dwpose_data = batch

        ### local image (first frame)
        image_local = []
        if 'local_image' in cfg.video_compositions:
            frames_num = misc_data.shape[1]
            bs_vd_local = misc_data.shape[0]
            image_local = misc_data[:,:1].clone().repeat(1, frames_num, 1, 1, 1)
            image_local_clone = rearrange(image_local, 'b f c h w -> b c f h w', b = bs_vd_local)
            image_local = rearrange(image_local, 'b f c h w -> b c f h w', b = bs_vd_local)
            if hasattr(cfg, "latent_local_image") and cfg.latent_local_image:
                with torch.no_grad():
                    temporal_length = frames_num
                    encoder_posterior = autoencoder.encode(video_data[:,0])
                    local_image_data = get_first_stage_encoding(encoder_posterior).detach()
                    image_local = local_image_data.unsqueeze(1).repeat(1,temporal_length,1,1,1) # [10, 16, 4, 64, 40]

        random_ref_frame = []
        if 'randomref' in cfg.video_compositions:
            random_ref_frame_clone = rearrange(random_ref_frame_data, 'b f c h w -> b c f h w')
            if hasattr(cfg, "latent_random_ref") and cfg.latent_random_ref:
                
                temporal_length = random_ref_frame_data.shape[1]
                encoder_posterior = autoencoder.encode(random_ref_frame_data[:,0].sub(0.5).div_(0.5))
                random_ref_frame_data = get_first_stage_encoding(encoder_posterior).detach()
                random_ref_frame_data = random_ref_frame_data.unsqueeze(1).repeat(1,temporal_length,1,1,1) # [10, 16, 4, 64, 40]

            random_ref_frame = rearrange(random_ref_frame_data, 'b f c h w -> b c f h w')


        if 'dwpose' in cfg.video_compositions:
            bs_vd_local = dwpose_data.shape[0]
            dwpose_data_clone = rearrange(dwpose_data.clone(), 'b f c h w -> b c f h w', b = bs_vd_local)
            if 'randomref_pose' in cfg.video_compositions:
                dwpose_data = torch.cat([random_ref_dwpose_data[:,:1], dwpose_data], dim=1)
            dwpose_data = rearrange(dwpose_data, 'b f c h w -> b c f h w', b = bs_vd_local)

        
        y_visual = []
        if 'image' in cfg.video_compositions:
            with torch.no_grad():
                vit_frame = vit_frame.squeeze(1)
                y_visual = clip_encoder.encode_image(vit_frame).unsqueeze(1) # [60, 1024]
                y_visual0 = y_visual.clone()
        
        # construct model inputs (CFG)
        full_model_kwargs=[{
            'y': None,
            "local_image": None if len(image_local) == 0 else image_local[:],
            'image': None if len(y_visual) == 0 else y_visual0[:],
            'dwpose': None if len(dwpose_data) == 0 else dwpose_data[:],
            'randomref': None if len(random_ref_frame) == 0 else random_ref_frame[:],
        }, 
                           {
            'y': None,
            "local_image": None, 
            'image': None,
            'randomref': None,
            'dwpose': None, 
        }]
        
        partial_keys = [
                ['image', 'randomref', "dwpose"],
            ]
        if hasattr(cfg, "partial_keys") and cfg.partial_keys:
            partial_keys = cfg.partial_keys


        opti_timesteps = getattr(cfg, 'opti_timesteps', cfg.Diffusion.schedule_param.num_timesteps)
        for partial_keys_one in partial_keys:
            model_kwargs = prepare_model_kwargs(partial_keys = partial_keys_one,
                                full_model_kwargs = full_model_kwargs,
                                use_fps_condition = cfg.use_fps_condition)

            batch_size, frames_num, _, _, _ = video_data.shape
            video_data = rearrange(video_data, 'b f c h w -> (b f) c h w')
            video_data_list = torch.chunk(video_data, video_data.shape[0]//cfg.chunk_size,dim=0)
            with torch.no_grad():
                decode_data = []
                for chunk_data in video_data_list:
                    latent_z = autoencoder.encode_firsr_stage(chunk_data, cfg.scale_factor).detach()
                    decode_data.append(latent_z) # [B, 4, 32, 56]
            video_data = torch.cat(decode_data,dim=0)
            video_data = rearrange(video_data, '(b f) c h w -> b c f h w', b = batch_size) # [B, 4, 16, 32, 56]

            t_round = torch.randint(0, opti_timesteps, (batch_size, ), dtype=torch.long, device=gpu) # 8

            # forward
            if cfg.use_fsdp:
                loss = diffusion.loss(x0=video_data, 
                    t=t_round, model=model, model_kwargs=model_kwargs,
                    use_div_loss=cfg.use_div_loss) 
                loss = loss.mean()
            else:
                with amp.autocast(enabled=cfg.use_fp16):
                    loss = diffusion.loss(
                            x0=video_data, 
                            t=t_round, 
                            model=model, 
                            model_kwargs=model_kwargs, 
                            use_div_loss=cfg.use_div_loss) # cfg.use_div_loss: False    loss: [80]
                    loss = loss.mean()

            # backward
            if cfg.use_fsdp:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_value_(model.parameters(), 0.05)
                optimizer.step()
            else:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            
            if not cfg.use_fsdp:
                scheduler.step()
            
            # ema update
            if cfg.use_ema:
                temp_state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
                for k, v in ema.items():
                    v.copy_(temp_state_dict[k].lerp(v, cfg.ema_decay))

            all_reduce(loss)
            loss = loss / cfg.world_size
            
            if cfg.rank == 0 and step % cfg.log_interval == 0: # cfg.log_interval: 100
                logging.info(f'Step: {step}/{cfg.num_steps} Loss: {loss.item():.3f} scale: {scaler.get_scale():.1f} LR: {scheduler.get_lr():.7f}')
                
        # Save checkpoint
        if step == cfg.num_steps or step % cfg.save_ckpt_interval == 0 or step == resume_step:
            os.makedirs(osp.join(cfg.log_dir, 'checkpoints'), exist_ok=True)
            if cfg.use_ema:
                local_ema_model_path = osp.join(cfg.log_dir, f'checkpoints/ema_{step:08d}_rank{cfg.rank:04d}.pth')
                save_dict = {
                    'state_dict': ema.module.state_dict() if hasattr(ema, 'module') else ema,
                    'step': step}
                torch.save(save_dict, local_ema_model_path)
                if cfg.rank == 0:
                    logging.info(f'Begin to Save ema model to {local_ema_model_path}')
            if cfg.rank == 0:
                local_model_path = osp.join(cfg.log_dir, f'checkpoints/non_ema_{step:08d}.pth')
                logging.info(f'Begin to Save model to {local_model_path}')
                save_dict = {
                    'state_dict': model.module.state_dict() if not cfg.debug else model.state_dict(),
                    'step': step}
                torch.save(save_dict, local_model_path)
                logging.info(f'Save model to {local_model_path}')

    if cfg.rank == 0:
        logging.info('Congratulations! The training is completed!')
    
    # synchronize to finish some processes
    if not cfg.debug:
        torch.cuda.synchronize()
        dist.barrier()


def prepare_model_kwargs(partial_keys, full_model_kwargs, use_fps_condition=False):
    
    if use_fps_condition is True:
        partial_keys.append('fps')

    partial_model_kwargs = [{}, {}]
    for partial_key in partial_keys:
        partial_model_kwargs[0][partial_key] = full_model_kwargs[0][partial_key]
        partial_model_kwargs[1][partial_key] = full_model_kwargs[1][partial_key]

    return partial_model_kwargs
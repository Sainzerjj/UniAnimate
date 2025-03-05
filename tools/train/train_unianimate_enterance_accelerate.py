import os
import os.path as osp
import sys
sys.path.insert(0, '/'.join(osp.realpath(__file__).split('/')[:-4]))
import json
import math
import einops
import random
import torch
import logging
import pynvml
from datetime import datetime, timedelta
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
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs
from accelerate import InitProcessGroupKwargs
from diffusers.optimization import get_scheduler
import mlflow
from tqdm.auto import tqdm

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
from IPython import embed

logger = get_logger(__name__, log_level="INFO")

@ENGINE.register_function()
def train_unianimate_entrance_accelerate(cfg_update,  **kwargs):
    for k, v in cfg_update.items():
        if isinstance(v, dict) and k in cfg:
            cfg[k].update(v)
        else:
            cfg[k] = v
    worker(cfg)
    return cfg

def worker(cfg):
    # kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    # process_group_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=5400)) 
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        mixed_precision=cfg.mixed_precision,
        log_with="mlflow",
        project_dir="./mlruns",
        # kwargs_handlers=[kwargs, process_group_kwargs],
    )

    current_time = datetime.now().strftime("%Y%m%d%H%M%S")    
    
    if cfg.use_fp16:
        weight_dtype = torch.float16
    else:
        weight_dtype = torch.float32

    # [Log] Save logging
    log_dir = generalized_all_gather(cfg.log_dir)[0]
    inf_name = osp.basename(cfg.cfg_file).split('.')[0]

    cfg.log_dir = osp.join(cfg.log_dir, inf_name)
    if accelerator.is_main_process:
        os.makedirs(osp.join(cfg.log_dir, 'checkpoints', current_time), exist_ok=True)

    # [Diffusion]  build diffusion settings
    diffusion = DIFFUSION.build(cfg.Diffusion)

    # [Dataset] imagedataset and videodataset
    len_frames = len(cfg.frame_lens) # 8  frame_lens: [16, 16, 16, 16, 16, 32, 32, 32]
    len_fps = len(cfg.sample_fps) # 8  sample_fps: [8,  8,  16, 16, 16, 8,  16, 16]
    cfg.max_frames = cfg.frame_lens[accelerator.local_process_index % len_frames]
    cfg.batch_size = cfg.batch_sizes[str(cfg.max_frames)] # 1 2
    cfg.sample_fps = cfg.sample_fps[accelerator.local_process_index % len_fps]
    
    # if accelerator.is_main_process:
    logging.info(f'Current worker {accelerator.local_process_index} with max_frames={cfg.max_frames}, batch_size={cfg.batch_size}, sample_fps={cfg.sample_fps}')
    
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

    train_trans_mask = data.Compose([
        data.Resize(cfg.resolution),
        data.ToTensor(),
        data.Normalize(mean=0.5, std=0.5)
        ])

    train_trans_vit = T.Compose([
                data.Resize(cfg.vit_resolution),
                T.ToTensor(),
                T.Normalize(mean=cfg.vit_mean, std=cfg.vit_std)])
    
    if cfg.max_frames == 1:
        cfg.sample_fps = 1
        dataset = DATASETS.build(cfg.img_dataset, transforms=train_trans, vit_transforms=train_trans_vit)
    else:
        dataset = DATASETS.build(cfg.vid_dataset, sample_fps=cfg.sample_fps, transforms=train_trans, vit_transforms=train_trans_vit, pose_transforms=train_trans_pose, mask_transforms=train_trans_mask, max_frames=cfg.max_frames)
    
    print(f"The dataset length is {dataset.__len__()}")

    distributed_sampler = DistributedSampler(
        dataset,
        num_replicas=accelerator.num_processes,
        shuffle=True,
        seed=cfg.seed
    )

    # DataLoaders creation:
    dataloader = DataLoader(
        dataset, 
        batch_size=cfg.batch_size,
        shuffle=False,
        sampler=distributed_sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        prefetch_factor=cfg.prefetch_factor)
    
    # [Model] embedder
    clip_encoder = EMBEDDER.build(cfg.embedder)
    clip_encoder.model.to(dtype=weight_dtype, device="cuda")
    with torch.no_grad():
        _, _, zero_y = clip_encoder(text="")

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
        logging.info('Load model from {} with status {} on step {}'.format(cfg.checkpoint_model, status, resume_step))
        
    model = model.to(device="cuda")

    whole_params_num = len(list(model.parameters())) # 1544

    # model.requires_grad_(False)

    if cfg.freezen_module:
        # 冻结时间模块和pose模块的参数
        # 750
        for name, module in model.named_modules():
            module_type_name = str(type(module))
            if 'Temporal' in module_type_name:
                # print(f'UnFreeze {name} module')
                for param in module.parameters():  
                        param.requires_grad = True
        # 30
        # embedding_names = [
        #     'dwpose_embedding',
        #     'dwpose_embedding_after',
        #     'randomref_pose2_embedding',
        #     'randomref_pose2_embedding_after'
        # ]
        # for name in embedding_names:
        #     if hasattr(model, name):
        #         embedding = getattr(model, name)
        #         for param in embedding.parameters():
        #             param.requires_grad = False

    trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    trainable_params_num = len(trainable_params)

    logging.info('The model has {} trainable parameters out of a total of {}'.format(trainable_params_num, whole_params_num))

    # [Model] auotoencoder 
    autoencoder = AUTO_ENCODER.build(cfg.auto_encoder).to(dtype=weight_dtype, device="cuda")
    autoencoder.eval() # freeze
    for param in autoencoder.parameters():
        param.requires_grad = False

    torch.cuda.empty_cache()

    if cfg.scale_lr:
        cfg.lr = (
            cfg.lr
            * cfg.gradient_accumulation_steps
            * cfg.batch_size
            * accelerator.num_processes
        )
    
    # optimizer
    if cfg.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = optim.AdamW
    
    trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = optimizer_cls(
        # model.parameters(),
        trainable_params,
        lr=cfg.lr,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        weight_decay=cfg.weight_decay,
        eps=cfg.adam_epsilon,
    )

    # scheduler
    scheduler = get_scheduler(
        cfg.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.warmup_steps
        * cfg.gradient_accumulation_steps,
        num_training_steps=cfg.num_steps
        * cfg.gradient_accumulation_steps,
    )

    (
        model,
        optimizer,
        dataloader,
        scheduler,
    ) = accelerator.prepare(
        model,
        optimizer,
        dataloader,
        scheduler,
    )

    num_update_steps_per_epoch = math.ceil(
        len(dataloader) / cfg.gradient_accumulation_steps
    )

    num_train_epochs = math.ceil(
        cfg.num_steps / num_update_steps_per_epoch
    )

    total_batch_size = (
        cfg.batch_size
        * accelerator.num_processes
        * cfg.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {cfg.batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(
        f"  Gradient Accumulation steps = {cfg.gradient_accumulation_steps}"
    )
    logger.info(f"  Total optimization steps = {cfg.num_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if cfg.resume_from_checkpoint:
        if cfg.resume_from_checkpoint != "latest":
            resume_dir = cfg.resume_from_checkpoint
        else:
            resume_dir = osp.join(cfg.log_dir, 'checkpoints', current_time)
        # Get the most recent checkpoint
        dirs = os.listdir(resume_dir)
        dirs = [d for d in dirs if d.startswith("checkpoint")]
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        path = dirs[-1]
        accelerator.load_state(os.path.join(resume_dir, path))
        accelerator.print(f"Resuming from checkpoint {path}")
        global_step = int(path.split("-")[1])

        first_epoch = global_step // num_update_steps_per_epoch
        resume_step = global_step % num_update_steps_per_epoch

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(global_step, cfg.num_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")
    
    for epoch in range(first_epoch, num_train_epochs):
        train_loss = 0.0
        print(f"The {epoch} of {num_train_epochs} Epoch in Training:")
        distributed_sampler.set_epoch(epoch)
        for step, batch in enumerate(dataloader):
            with accelerator.accumulate(model):
                vit_frame, video_data, hand_masks_data, misc_data, dwpose_data, random_ref_frame_data, random_ref_dwpose_data = [item.to(dtype=weight_dtype) for item in batch]
                ### local image (first frame)

                image_local = []
                if 'local_image' in cfg.video_compositions:
                    frames_num = misc_data.shape[1]
                    bs_vd_local = misc_data.shape[0]
                    image_local = misc_data[:,:1].clone().repeat(1, frames_num, 1, 1, 1)
                    image_local = rearrange(image_local, 'b f c h w -> b c f h w', b = bs_vd_local)
                    if hasattr(cfg, "latent_local_image") and cfg.latent_local_image:
                        with torch.no_grad():
                            temporal_length = frames_num
                            encoder_posterior = autoencoder.encode(video_data[:,0])
                            local_image_data = get_first_stage_encoding(encoder_posterior).detach()
                            image_local = local_image_data.unsqueeze(1).repeat(1,temporal_length,1,1,1) # [10, 16, 4, 64, 40]

                random_ref_frame = []
                if 'randomref' in cfg.video_compositions:
                    if hasattr(cfg, "latent_random_ref") and cfg.latent_random_ref:
                        temporal_length = random_ref_frame_data.shape[1]

                        # noise_aug_strength = rand_log_normal(
                        #     shape=[video_data.shape[0], 1, 1, 1], 
                        #     loc=-3.0, 
                        #     scale=0.5
                        # ).to(video_data.device).to(weight_dtype)
                        # random_ref_frame_data = random_ref_frame_data[:,0].sub(0.5).div_(0.5)
                        # random_ref_frame_data = random_ref_frame_data + noise_aug_strength * torch.randn_like(random_ref_frame_data)
                        encoder_posterior = autoencoder.encode(random_ref_frame_data[:,0].sub(0.5).div_(0.5))
                        random_ref_frame_data = get_first_stage_encoding(encoder_posterior).detach()
                        random_ref_frame_data = random_ref_frame_data.unsqueeze(1).repeat(1,temporal_length,1,1,1) # [10, 16, 4, 64, 40]

                    random_ref_frame = rearrange(random_ref_frame_data, 'b f c h w -> b c f h w')


                if 'dwpose' in cfg.video_compositions:
                    bs_vd_local = dwpose_data.shape[0]
                    if 'randomref_pose' in cfg.video_compositions:
                        dwpose_data = torch.cat([random_ref_dwpose_data[:,:1], dwpose_data], dim=1)
                    dwpose_data = rearrange(dwpose_data, 'b f c h w -> b c f h w', b = bs_vd_local)

                y_visual = []
                if 'image' in cfg.video_compositions:
                    with torch.no_grad():
                        vit_frame = vit_frame.squeeze(1)
                        y_visual = clip_encoder.encode_image(vit_frame).unsqueeze(1) # [60, 1024]
                        y_visual0 = y_visual.clone()
                
                # construct model cond
                model_kwargs={
                    'y': None,
                    "local_image": None if len(image_local) == 0 else image_local[:],
                    'image': None if len(y_visual) == 0 else y_visual0[:],
                    'dwpose': None if len(dwpose_data) == 0 else dwpose_data[:],
                    'randomref': None if len(random_ref_frame) == 0 else random_ref_frame[:],
                }
                prob = torch.rand(1)
                if prob > 1 - cfg.p_zero:
                    pass
                elif prob < cfg.p_zero:
                    model_kwargs = {key: (torch.zeros_like(value) if value is not None else None) for key, value in model_kwargs.items()}
                else:
                    if torch.rand(1).item() > 0.5:
                        model_kwargs['local_image'] = torch.zeros_like(model_kwargs['local_image'])
                    if torch.rand(1).item() > 0.5:
                        model_kwargs['randomref'] = torch.zeros_like(model_kwargs['randomref'])
                    if torch.rand(1).item() > 0.5:
                        model_kwargs['image'] = torch.zeros_like(model_kwargs['image'])
                
                partial_keys = [
                        ['image', 'randomref', "dwpose"],
                ]
                if hasattr(cfg, "partial_keys") and cfg.partial_keys:
                    partial_keys = cfg.partial_keys


                opti_timesteps = getattr(cfg, 'opti_timesteps', cfg.Diffusion.schedule_param.num_timesteps)
                for partial_key in partial_keys:
                    model_kwargs = prepare_model_kwargs(partial_keys = partial_key,
                                        full_model_kwargs = model_kwargs,
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

                    t_round = torch.randint(0, opti_timesteps, (batch_size, ), dtype=torch.long, device=video_data.device) # 8

                    if cfg.use_hand_mask_loss:
                        hand_masks_data = einops.rearrange(
                            hand_masks_data, "b f c h w -> (b f) c h w"
                        )
                        hand_masks_data = torch.nn.functional.max_pool2d(hand_masks_data, 8, 8) * 0.5 + 0.5
                        hand_masks_data = einops.rearrange(hand_masks_data, "(b f) c h w -> b f c h w", b=cfg.batch_size)
                        hand_masks_data = 1 + hand_masks_data * 9

                    # forward
                    if cfg.use_fsdp:
                        loss = diffusion.loss(x0=video_data, 
                            t=t_round, model=model, model_kwargs=model_kwargs, use_div_loss=cfg.use_div_loss) 
                        loss = loss.mean()
                    else:
                        with amp.autocast(enabled=cfg.use_fp16):
                            loss = diffusion.loss(
                                    x0=video_data, 
                                    t=t_round, 
                                    model=model, 
                                    model_kwargs=model_kwargs, 
                                    use_div_loss=cfg.use_div_loss, # cfg.use_div_loss: False    loss: [80]
                                    loss_mask=hand_masks_data
                                ) 
                            loss = loss.mean()
                    
                    # Gather the losses across all processes for logging (if we use distributed training).
                    avg_loss = accelerator.gather(loss.repeat(cfg.batch_size)).mean()
                    train_loss += avg_loss.item() / cfg.gradient_accumulation_steps

                    # backward
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(
                            trainable_params,
                            cfg.max_grad_norm,
                        )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0
                if global_step % cfg.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = osp.join(cfg.log_dir, 'checkpoints', current_time, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
            
            logs = {
                "step_loss": loss.detach().item(),
                "lr": scheduler.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)

            if global_step % cfg.save_ckpt_interval == 0 and accelerator.is_main_process:
                unwrap_model = accelerator.unwrap_model(model)
                local_model_path = osp.join(cfg.log_dir, f'checkpoints/{current_time}/non_ema_{epoch}_{global_step:08d}.pth')
                logging.info(f'Begin to Save model to {local_model_path}')
                save_dict = {
                    'state_dict': unwrap_model.state_dict(),
                    'step': step
                }
                torch.save(save_dict, local_model_path)
                logging.info(f'Save model to {local_model_path}')

            if global_step >= cfg.num_steps:
                break
                    
        if epoch % cfg.save_model_epoch_interval == 0 and accelerator.is_main_process:
            unwrap_model = accelerator.unwrap_model(model)
            local_model_path = osp.join(cfg.log_dir, f'checkpoints/{current_time}/non_ema_{epoch}_{global_step:08d}.pth')
            logging.info(f'Begin to Save model to {local_model_path}')
            save_dict = {
                'state_dict': unwrap_model.state_dict(),
                'step': step
            }
            torch.save(save_dict, local_model_path)
            logging.info(f'Save model to {local_model_path}')
                
    if accelerator.is_main_process:
        logging.info('Congratulations! The training is completed!')

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    accelerator.end_training()
    

def prepare_model_kwargs(partial_keys, full_model_kwargs, use_fps_condition=False):
    
    if use_fps_condition is True:
        partial_keys.append('fps')

    partial_model_kwargs = {}
    for partial_key in partial_keys:
        partial_model_kwargs[partial_key] = full_model_kwargs[partial_key]

    return partial_model_kwargs

def rand_log_normal(
    shape, 
    loc=0., 
    scale=1., 
    device='cpu', 
    dtype=torch.float32,
    generator=None
):
    """Draws samples from an lognormal distribution."""
    # u = torch.rand(shape, dtype=dtype, device=device, generator=generator) * (1 - 2e-7) + 1e-7
    rnd_normal = torch.randn(
            shape, device=device, dtype=dtype, generator=generator) # N(0, I)
    sigma = (rnd_normal * scale + loc).exp()
    # return torch.distributions.Normal(loc, scale).icdf(u).exp()
    return sigma
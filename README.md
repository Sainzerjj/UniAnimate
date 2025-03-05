
<!-- main documents -->
# UniAnimate: Taming Unified Video Diffusion Models for Consistent Human Image Animation
The unofficial implementation of the paper "UniAnimate: Taming Unified Video Diffusion Models for Consistent Human Image Animation." The project is continuously being optimized.


## TODO
- [x] The training code for the models on higher resolution videos.
- [ ] The training code for the models based on VideoLCM for faster video synthesis.



## Introduction

<div align="center">
<p align="middle">
  <img src='https://img.alicdn.com/imgextra/i3/O1CN01VvncFJ1ueRudiMOZu_!!6000000006062-2-tps-2654-1042.png' width='784'>

  Overall framework of UniAnimate
</p>
</div>

Recent diffusion-based human image animation techniques have demonstrated impressive success in synthesizing videos that faithfully follow a given reference identity and a sequence of desired movement poses. Despite this, there are still two limitations: i) an extra reference model is required to align the identity image with the main video branch, which significantly increases the optimization burden and model parameters; ii) the generated video is usually short in time (e.g., 24 frames), hampering practical applications. To address these shortcomings, we present a UniAnimate framework to enable efficient and long-term human video generation. First, to reduce the optimization difficulty and ensure temporal coherence, we map the reference image along with the posture guidance and noise video into a common feature space by incorporating a unified video diffusion model. Second, we propose a unified noise input that supports random noised input as well as first frame conditioned input, which enhances the ability to generate long-term video. Finally, to further efficiently handle long sequences, we explore an alternative temporal modeling architecture based on state space model to replace the original computation-consuming temporal Transformer. Extensive experimental results indicate that UniAnimate achieves superior synthesis results over existing state-of-the-art counterparts in both quantitative and qualitative evaluations. Notably, UniAnimate can even generate highly consistent one-minute videos by iteratively employing the first frame conditioning strategy.


## Getting Started with UniAnimate


### (1) Installation

Installation the python dependencies:

```
git clone https://github.com/ali-vilab/UniAnimate.git
cd UniAnimate
conda create -n UniAnimate python=3.9
conda activate UniAnimate
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```
We also provide all the dependencies in `environment.yaml`. 

**Note**: for Windows operating system, you can refer to [this issue](https://github.com/ali-vilab/UniAnimate/issues/11) to install the dependencies. Thanks to [@zephirusgit](https://github.com/zephirusgit) for the contribution. If you encouter the problem of `The shape of the 2D attn_mask is torch.Size([77, 77]), but should be (1, 1).`, please refer to [this issue](https://github.com/ali-vilab/UniAnimate/issues/61) to solve it, thanks to [@Isi-dev](https://github.com/Isi-dev) for the contribution.

### (2) Download the pretrained checkpoints

Download models:
```
!pip install modelscope
from modelscope.hub.snapshot_download import snapshot_download
model_dir = snapshot_download('iic/unianimate', cache_dir='checkpoints/')
```
Then you might need the following command to move the checkpoints to the "checkpoints/" directory:
```
mv ./checkpoints/iic/unianimate/* ./checkpoints/
```

Finally, the model weights will be organized in `./checkpoints/` as follows:
```
./checkpoints/
|---- dw-ll_ucoco_384.onnx
|---- open_clip_pytorch_model.bin
|---- unianimate_16f_32f_non_ema_223000.pth 
|---- v2-1_512-ema-pruned.ckpt
└---- yolox_l.onnx
```

### (3) Pose alignment **(Important)**

Please refer to the official repository's [Link](https://github.com/ali-vilab/UniAnimate) instructions.


### (4) Run the UniAnimate model to generate videos

#### (4.1) Generating video clips (32 frames with 768x512 resolution)

Execute the following command to generate video clips:
```
python inference.py --cfg configs/UniAnimate_infer.yaml 
```
After this, 32-frame video clips with 768x512 resolution will be generated:


**<font color=red>&#10004; Some tips</font>**:

- > To run the model, **~12G** ~~26G~~ GPU memory will be used. If your GPU is smaller than this, you can change the  `max_frames: 32` in `configs/UniAnimate_infer.yaml` to other values, e.g., 24, 16, and 8. Our model is compatible with all of them.


#### (4.2) Generating video clips (32 frames with 1216x768 resolution)

If you want to synthesize higher resolution results, you can change the `resolution: [512, 768]` in `configs/UniAnimate_infer.yaml` to `resolution: [768, 1216]`. And execute the following command to generate video clips:
```
python inference.py --cfg configs/UniAnimate_infer.yaml 
```
After this, 32-frame video clips with 1216x768 resolution will be generated:


**<font color=red>&#10004; Some tips</font>**:

- > To run the model, **~21G** ~~36G~~ GPU memory will be used.  Even though our model was trained on 512x768 resolution, we observed that direct inference on 768x1216 is usually allowed and produces satisfactory results. If this results in inconsistent apparence, you can try a different seed or adjust the resolution to 512x768.

- > Although our model was not trained on 48 or 64 frames, we found that the model generalizes well to synthesis of these lengths.



In the `configs/UniAnimate_infer.yaml` configuration file, you can specify the data, adjust the video length using `max_frames`, and validate your ideas with different Diffusion settings, and so on.



#### (4.3) Generating long videos

If you want to synthesize videos as long as the target pose sequence, you can execute the following command to generate long videos:
```
python inference.py --cfg configs/UniAnimate_infer_long.yaml
```
After this, long videos with 1216x768 resolution will be generated:


In the `configs/UniAnimate_infer_long.yaml` configuration file, `test_list_path` should in the format of `[frame_interval, reference image, driving pose sequence]`, where `frame_interval=1` means that all frames in the target pose sequence will be used to generate the video, and `frame_interval=2` means that one frame is sampled every two frames. `reference image` is the location where the reference image is saved, and `driving pose sequence` is the location where the driving pose sequence is saved.





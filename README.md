
<!-- main documents -->
# UniAnimate: Taming Unified Video Diffusion Models for Consistent Human Image Animation
The unofficial implementation of the paper "UniAnimate: Taming Unified Video Diffusion Models for Consistent Human Image Animation." **The project is continuously being optimized...**


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
git clone https://github.com/Sainzerjj/UniAnimate.git
cd UniAnimate
conda create -n UniAnimate python=3.9
conda activate UniAnimate
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```

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
### (3) Prepare Dataset

Modify the `vid_dataset` variable in `configs/UniAnimate_train_accelerate.yaml` to prepare the training dataset. The content of the JSON file `data/dance_dataset.json` consists of the training video and corresponding pose video paths, which would be read in the Dataset Class `UniAnimateDataset`, defined in `tools/dataset/unianimate_dataset.py`.

### (4) Training the UniAnimate
Execute the following command to start training:
```
bash train_accelerate.sh
```

### (5) Pose alignment

Rescale the target pose sequence to match the pose of the reference image. We modify the official alignment code to optimize the overall human body effect. Execute the following command for pose alignment.
```
bash run_align_pose.sh
```

### (6) Run the UniAnimate model to generate videos

#### (1) Generating video clips

Execute the following command to generate video clips:
```
bash inference.sh
# --cfg configs/UniAnimate_infer.yaml 
```
If you want to synthesize higher resolution results, you can change the default `resolution: [512, 768]` in `configs/UniAnimate_infer.yaml` to `resolution: [768, 1216]`. And execute the above command to generate video clips.

#### (2) Generating long videos

If you want to synthesize videos as long as the target pose sequence, you can execute the following command to generate long videos:
```
bash inference.sh
# --cfg configs/UniAnimate_infer_long.yaml
```
After this, long videos with the pre-set resolution will be generated.






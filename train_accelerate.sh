CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" accelerate launch --main_process_port=29501 \
--config_file configs/ds_config.yaml \
scripts/train.py --cfg configs/UniAnimate_train_accelerate.yaml > UniAnimate_log.txt 2>&1 &
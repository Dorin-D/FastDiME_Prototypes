#!/bin/bash -l 

# 256x256 uncond: 
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
MODELPATH="/home/dorin/Research/repos/models/imagenet_cond_ddpm/256x256_diffusion_uncond.pt"

SAMPLE_FLAGS="--batch_size 50 --timestep_respacing 200"
# DATAPATH="/home/dorin/Research/repos/github/DiME_CelebA/CelebA/Img"
DATAPATH="dataset_256.yml"
ORACLEPATH="DiME_Models/oracle.pth"
OUTPUT_PATH="output/004_AttemptNewDDPM"
EXPNAME="004_Attempt_256_256"
# CLASSIFIERPATH="DiME_Models/classifier.pth"
# CABRNET=0
CLASSIFIERPATH="DiME_Models/protopnet_resnet_256_cub.pth"
CABRNET=1
MODEL_ARCH="DiME_Models/model_arch.yml"
TARGET_CLASS=37


# parameters of the sampling
GPU=0
S=60
SEED=4
USE_LOGITS=True
CLASS_SCALES='8,10,15'
LAYER=18
PERC=30
L1=0.05
QUERYLABEL=31
TARGETLABEL=-1
IMAGESIZE=256  # dataset shape
NUMBATCHES=50
# CHUNK=0
NUM_CHUNKS=20
SUBSAMPLING=30

source ~/miniforge3/etc/profile.d/conda.sh
conda activate cabrnet


python -W ignore main.py $MODEL_FLAGS $SAMPLE_FLAGS \
    --query_label $QUERYLABEL --target_label $TARGETLABEL \
    --output_path $OUTPUT_PATH --num_batches $NUMBATCHES \
    --start_step $S --dataset 'Cub200' --data_dir $DATAPATH \
    --exp_name $EXPNAME --gpu $GPU \
    --model_path $MODELPATH --classifier_scales $CLASS_SCALES \
    --classifier_path $CLASSIFIERPATH --seed $SEED \
    --oracle_path $ORACLEPATH \
    --l1_loss $L1 --use_logits $USE_LOGITS \
    --l_perc $PERC --l_perc_layer $LAYER \
    --save_x_t True --save_z_t True \
    --use_sampling_on_x_t True \
    --save_images True --image_size $IMAGESIZE --chunk 0 --num_chunks $NUM_CHUNKS \
    --subsampling $SUBSAMPLING --cabrnet $CABRNET --model_arch $MODEL_ARCH \
    --target_class $TARGET_CLASS
#!/bin/bash
CUDA_VISIBLE_DEVICES=1 \
python test.py \
    --batchSize 4 \
    --netG dyconvcontext4coco \
    --checkpoints_dir checkpoints \
    --which_epoch 200 \
    --name ade_best \
    --dataset_mode ade20k \
    --dataroot datasets/ade20k \
    --use_vae

#!/bin/bash
python test.py \
    --batchSize 4 \
    --netG condconv \
    --checkpoints_dir checkpoints \
    --which_epoch 200 \
    --name cs_best \
    --dataset_mode cityscapes \
    --dataroot datasets/cityscapes \
    --use_vae

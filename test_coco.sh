#!/bin/bash
python test.py \
    --batchSize 4 \
    --netG condconv \
    --checkpoints_dir checkpoints \
    --which_epoch 100 \
    --name coco_best \
    --dataset_mode coco \
    --dataroot datasets/coco_stuff \
    --use_vae

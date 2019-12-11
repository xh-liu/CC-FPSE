ip="$(ifconfig | egrep -A 1 "enp129s0f0|eth0" | tail -1 | cut -d ':' -f 2 | cut -d ' ' -f 1)"
python train.py \
    --name coco_cc_fpse \
    --mpdist \
    --netG condconv \
    --dist_url tcp://$ip:8000 \
    --num_servers 1 \
    --netD fpse \
    --lambda_feat 20 \
    --dataset_mode coco \
    --dataroot datasets/coco_stuff \
    --batchSize 1 \
    --niter 100 \
    --niter_decay 100 \
    --use_vae

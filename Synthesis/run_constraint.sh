#!/bin/bash

python main.py -g 1 \
--num_start=5 --num_itr=75000 --num_gen=1 \
--rdir='../data/GLAS/images' \
--ref_dir='fg,bg' \
--seg_ext='anno2.bmp' \
--gen_type='constraint' \
--gen_seg_dir='../data/GLAS/gen_masks' \
--pretrained_file='./vgg16.npy' \
--lr=1e-3 \
--alpha=1e-3 \
--beta=10.0 \
--gamma=10.0 \
--gen_w=775 --gen_h=522
#--gen_w=512 --gen_h=512





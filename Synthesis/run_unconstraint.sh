#!/bin/bash

python main.py -g 0 \
--num_start=0 --num_itr=30000 --num_gen=1 \
--rdir='../data/mixtures' \
--ref_dir='ADU,UO4' \
--seg_ext='' \
--gen_type='unconstraint' \
--gen_seg_dir='' \
--pretrained_file='./vgg16_nuclear96.npy' \
--lr=1e-3 \
--alpha=1.0 \
--beta=0.0 \
--gamma=10.0 \
--gen_w=512 --gen_h=512





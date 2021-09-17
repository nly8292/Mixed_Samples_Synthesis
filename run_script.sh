#!/bin/bash

python main.py --gpu_id 0 \
--root_dir='/usr/sci/projs/DeepLearning/Cuong_Dataset/Nuclear_Forensics_Data/Mixtures/Mixtures_Texture_Preprocessed/Train_Manual_Preprocessed' \
--precursors='ADU','MDU','SDU' \
--w_precursors=0.0,1.0,0.0 \
--start_num=0 --num_gen=15 \
--num_scale=2 --itr_vec=10000,10000 \
--extracted_layers='c11r','c21r','c31r','c41r','c51r' \
--lr=1e-3 \
--w_stat=1.0 --w_tv=1.0 \
--gen_w=128 --gen_h=128 \
--transf_type=0 \
--opt_type='Adam' \
--stat_type='gram' \
--display_ref_gen

    
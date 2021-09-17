#!/home/sci/nly8292/document/pytorch3/bin/python

#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH -o slurm-%j.out-%N
#SBATCH -e slurm-%j.err-%N
#SBATCH  --gres=gpu:1

from __future__ import print_function, division, absolute_import, unicode_literals
import numpy as np
import os
import glob
import argparse
import sys
sys.path.append(os.getcwd())
import logging
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import torch
import torchvision
from torchvision import datasets, models, transforms

from dataset import *
from model import *
from utils import *

def main():
    
    ## Parse input from command prompt ##
    parser = argparse.ArgumentParser()       
    parser.add_argument('--gpu_id', type=int, default = 0, help = 'GPU#')  
    parser.add_argument('--num_scale', type=int, default = 3, help = 'Number of downsampled input')
    parser.add_argument('--itr_vec', type=str, default = [1000,1000,1000], help = 'Number of iters at each scale')   
    parser.add_argument('--lr', type=float, default = 1e-3, help = 'Learning rate')               
    parser.add_argument('--start_num', type=int, default = 0, help = 'Starting index of generations')
    parser.add_argument('--num_gen', type=int, default = 0, help = 'Number of generated images')
    parser.add_argument('--precursors', type=str, default = [], help = 'Subdir of precursors used for generating mixed samples')
    parser.add_argument('--transf_type', type=int, default = 0, help = 'Augmentation method')
    parser.add_argument('--w_precursors', type=str, default = [], help = 'The contribution of each precursor to the generated images')
    parser.add_argument('--extracted_layers', type=str, default = ['c11r','c21r','c31r','c41r','c51r'], help = 'Extracted layers from VGG16 for computing statistics')
    parser.add_argument('--w_stat', type=float, default = 1.0, help = 'Weighted param for statistics loss')
    parser.add_argument('--w_tv', type=float, default = 1.0, help = 'Weighted param for TV loss')    
    parser.add_argument('--gen_w', type=int, default = 256, help = 'Width of generated image')
    parser.add_argument('--gen_h', type=int, default = 256, help = 'Height of generated image')
    parser.add_argument('--opt_type', type=str, default = 'Adam', help = 'Optimizer (Adam|LBFGS)')
    parser.add_argument('--stat_type', type=str, default = 'gram', help = 'The statistic to be computed for the extracted features')
    parser.add_argument('--root_dir', type=str, default = './', help = 'Root directory of reference')
    parser.add_argument('--out_dir', type=str, default = './outputs', help = 'Directory for storing generated images')
    parser.add_argument('--tensorboard_dir', type=str, default = './tensorboard', help = 'Directory for storing tensorboard results')
    parser.add_argument("--log_tensorboard", action="store_true", help = 'Turn on tensorboard')
    parser.add_argument("--display_ref_gen", action="store_true", help = 'Displaying reference images and generated image on the same canvas')
               
    params, unparsed = parser.parse_known_args()
    
    if len(params.precursors) == 0:
        print('No precursors subdirectory available!')
        return

    ## Convert input strings into a list ##
    params.itr_vec = [int(x) for x in params.itr_vec.split(',')]
    params.precursors = [x for x in params.precursors.split(',')]
    params.extracted_layers = [x for x in params.extracted_layers.split(',')]
    if len(params.w_precursors) > 0:
        params.w_precursors = [float(x) for x in params.w_precursors.split(',')]
        if len(params.w_precursors) != len(params.precursors):
            print('Mismatch between the number of precursors and the weight of each precursor!')
            return
        
    
    logger = create_logger()
    ## Logging params ##
    log_params(logger,params)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(params.gpu_id)       

    data_loader = []
    for p in params.precursors:
        p = '100_%s' %p
        if 'UO4' in params.precursors:
            p += '_60k'
        curr_dir = '%s/%s' %(params.root_dir, p)
        curr_p = Mixtures(curr_dir, params.transf_type)
        data_loader.append(curr_p)        

    print("Finished loading data!\n") 
    print("Utilize {:} optimizer".format(params.opt_type))   
    
    syn_model = Model(params, logger)
    syn_model.optimize(data_loader, params.opt_type, gen_w=params.gen_w, gen_h=params.gen_h)

if __name__ == '__main__':
    main()
    
            


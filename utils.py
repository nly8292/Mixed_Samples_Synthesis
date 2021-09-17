from __future__ import print_function, division, absolute_import, unicode_literals
from PIL import Image
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import datetime
import logging
import logging.config

import torch
import torchvision.utils as vutils

def save_img(out, out_name, isHist=False, isGray=False): 
    '''
    Save given tensor as an image file
    '''

    gen_tensor = np.copy(np.squeeze(out,0))
                                 
    curr_gen = np.transpose( gen_tensor, (1,2,0))    
    curr_gen *= 255

    if isGray:
        curr_gen = np.mean(curr_gen,axis=2,keepdims=True)
        curr_gen = np.tile(curr_gen,(1,1,3))            
    curr_gen = np.clip(curr_gen, 0, 255).astype('uint8')

    img = Image.fromarray(curr_gen, mode='RGB')          
    img.save(out_name)

    if isHist:
        if not os.path.exists('./hist/'):
            os.makedirs('./hist/')

        cl = ['r', 'g', 'b']
        sym = ['--','-*','-^','-+']

        style_img = (style*255).astype('uint8')
        fg,ax = plt.subplots(3,1)
        for i in range(3):        
            histx = cv2.calcHist([curr_gen], [i], None, [256], [0,256])     
            ax[i].plot(histx/np.max(histx),cl[i])                               
            histy = cv2.calcHist([style_img], [i], None, [256], [0,256])           
            ax[i].plot(histy/np.max(histy),cl[i]+sym[0])    

        plt.tight_layout()
        plt.savefig('./hist_%s.png' %out_name.split('/')[-1][:-4])

def save_real_gen(ref_img_list, gen_img, out_name, mu, std, transf_type):
    '''
    Save reference images and generated image on the same canvas
    '''

    viz_r = ref_img_list[0]*std + mu
    for rx in ref_img_list[1:]:
        viz_r = torch.cat((viz_r,rx*std + mu),0)    
    viz_r = viz_r.cpu()
    
    viz_g = torch.zeros(viz_r.size())    
    if transf_type == 1:
        viz_g[len(ref_img_list)//2:(len(ref_img_list)//2)+1,...] \
            = torch.sigmoid(gen_img.detach())
    else:
        viz_g[len(ref_img_list)//2:(len(ref_img_list)//2)+1,...] \
            = torch.clamp(gen_img.detach(),0.0,1.0)
    viz_g = torch.mean(viz_g, 1, keepdim=True)
    viz_g = viz_g.repeat(1,3,1,1)                    
    
    viz = torch.cat((viz_r,viz_g),0)                    
    
    viz = np.transpose( vutils.make_grid(viz, nrow=len(ref_img_list)).numpy(), (1,2,0))                    
    img = Image.fromarray((viz*255).astype('uint8'), mode='RGB')                                    
    img.save(out_name[:-4] + '_viz.png')
                
def create_logger():
    ''' 
    Create logger object
    '''

    curr = datetime.datetime.now()
    if not os.path.exists('./logs'):
        os.makedirs('./logs')
    
    logger_name = './logs/%s_info.log'\
                    %(str(curr.year)+'_'+str(curr.month)+'_'\
                        +str(curr.day)+'_'+str(curr.hour)+\
                        str(curr.minute)+str(curr.second))
    
    logger = logging.getLogger(logger_name.split('/')[-1][:-4])
    logger.setLevel(logging.INFO)
    
    fh = logging.FileHandler(logger_name); fh.setLevel(logging.INFO)    
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)

    logger.addHandler(fh)
    
    return logger

def log_params(logger,params):
    '''
    Log user params
    '''

    logger.info('##### Params #####')
    logger.info('Root Dir:\t {:}'.format(params.root_dir))
    logger.info('Precursors:\t {:}'.format(params.precursors))
    logger.info('Weight of precursors:\t {:}'.format(params.w_precursors))
    logger.info('Number of generated images:\t {:}'.format(params.num_gen))
    logger.info('Starting index for generated images:\t {:}\n'.format(params.start_num))    
    logger.info('Number of downsampled inputs:\t {:}'.format(params.num_scale))    
    logger.info('Number of iters at each scale: \t{:}'.format("".join(str(params.itr_vec))))
    logger.info('Extracted layers:\t {:}'.format(params.extracted_layers))
    logger.info('Augmentation method: \t{:}'.format(params.transf_type))
    logger.info('Generated Images Dim: \tW = {:} - H = {:}'.format(params.gen_w,params.gen_h))
    logger.info('Type of Statistic: \t{:}'.format(params.stat_type))
    logger.info('Optimization Method: \t{:}'.format(params.opt_type))
    logger.info('Learning rate: \t{:}'.format(params.lr))
    logger.info('Weight for statistic loss: \t{:}'.format(params.w_stat))
    logger.info('Weight for TV loss: \t{:}'.format(params.w_tv))
    logger.info('##########\n')

    
        
    

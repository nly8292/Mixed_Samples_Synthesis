import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.autograd import Variable

LAYERS_IND = {'c11':0, 'c11r':1, 'c12':2, 'c12r':3, 'pool1':4,
                'c21':5, 'c21r':6, 'c22':7, 'c22r':8, 'pool2':9,
                'c31':10, 'c31r':11, 'c32':12, 'c32r':13, 'c33':14, 'c33r':15, 'pool3':16,
                'c41':17, 'c41r':18, 'c42':19, 'c42r':20, 'c43':21, 'c43r':22, 'pool4':23,
                'c51':24, 'c51r':25, 'c52':26, 'c52r':27, 'c53':28, 'c53r':29, 'pool5':30}

class Net(nn.Module):
    '''
    Set up pretrained VGG16 for extracting desired layers
    '''

    def __init__(self, extracted_layers=['c11r','c21r','c31r','c41r','c51r']):

        super(Net, self).__init__()          
        
        model_class = models.vgg16(pretrained=True)                    
        model_class = model_class.features                
        
        ## Put layers in between extracted outputs into a sequential block ##
        self.network = nn.ModuleList( [nn.Sequential(*model_class[:LAYERS_IND[extracted_layers[0]]+1])] )
        for li,layer in enumerate(extracted_layers[1:], start=1):
            self.network.append( nn.Sequential(*model_class[LAYERS_IND[extracted_layers[li-1]]+1:LAYERS_IND[layer]+1]) )

        print('Sequential Blocks for layer extraction:')
        for l in self.network:
            print(l)

        self.extracted_layers = extracted_layers
        
    def forward(self, x):
        ftrs_dict = {}
        ftrs_dict[self.extracted_layers[0]] = self.network[0](x)        
        for li,layer in enumerate(self.extracted_layers[1:], start=1):            
            ftrs_dict[layer] = self.network[li](ftrs_dict[self.extracted_layers[li-1]])
        
        return ftrs_dict

class Stats(nn.Module):
    '''
    Compute specific statistics for the extracted layer
    '''

    def __init__(self, stat_type='gram'):

        super(Stats, self).__init__()   
        self.stat_type = stat_type

    def forward(self, ftrs_dict):

        ftrs_list = ftrs_dict.values()          
        ## Compute statistics of each extracted layer ##        
        gram_list = self._agg_stats(ftrs_list)   
                
        return gram_list, ftrs_list

    def _agg_stats(self, ftrs_list):
        '''
        Put the computed statistics into a list
        '''

        out_list = []                
        for f in ftrs_list:   
            rs_tensor = torch.squeeze(f,0).reshape(f.size(1), f.size(2)*f.size(3))      
            out_list.append( self._compute_stat(rs_tensor) )                              
                
        return out_list

    def _compute_stat(self, tensor):
        '''
        Compute specific statistic of a given tensor
        '''

        if self.stat_type == 'mean':            
            out = torch.mean(tensor, axis=1, keepdim=True)
        elif self.stat_type == 'gram':
            ## Compute Gram ##
            out = torch.matmul(tensor, torch.transpose(tensor,1,0)) / tensor.size(1)           
        elif self.stat_type == 'cov':
            ## Covariance ##
            g = torch.matmul(tensor, torch.transpose(tensor,1,0)) / tensor.size(1)        
            mu = torch.mean(rs_tensor, axis=1, keepdim=True)
            mumu = torch.matmul(mu,mu.T)
            out = g - mumu                                              

        return out

class LossModel(nn.Module):
    '''
    Compute the loss between generated and reference samples
    '''

    def __init__(self, loss_type='MSE'):

        super(LossModel, self).__init__()
        if loss_type == 'MSE':
            self.dif = nn.MSELoss()
        else:
            self.dif = nn.L1Loss()

    def forward(self, gen, ref):  

        loss = self.dif(gen[0],ref[0]) 
        for g, r in zip(gen[1:], ref[1:]):            
            loss += (self.dif(g,r))
        
        return loss

class LossTV(nn.Module):
    '''
    Compute TV loss
    '''

    def __init__(self):

        super(LossTV,self).__init__()

    def forward(self,img):

        bs_img, c_img, h_img, w_img = img.size()
        tv_h = torch.pow(img[:,:,1:,:]-img[:,:,:-1,:], 2).sum()
        tv_w = torch.pow(img[:,:,:,1:]-img[:,:,:,:-1], 2).sum()
        return (tv_h+tv_w)/(bs_img*c_img*h_img*w_img)

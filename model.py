import os
import numpy as np
import time
import matplotlib.pyplot as plt
import random

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import torchvision.utils as vutils

import network
from utils import *


class Model(object):

    def __init__(self, params, logger):

        self.logger = logger
        self.precursors = params.precursors
        self.w_precursors = params.w_precursors
        self.lr = params.lr  
        self.num_gen = params.num_gen; self.start_num = params.start_num
        self.w_stat = params.w_stat; self.w_tv = params.w_tv
        self.transf_type = params.transf_type
        self.log_tensorboard = params.log_tensorboard 
        self.display_ref_gen = params.display_ref_gen 
        
        self.num_scale = params.num_scale
        self.itr_vec = params.itr_vec
        if len(params.itr_vec) != params.num_scale:
            print('Mismatch in total number of scales! Set number of iters at each scale to 1000!')
            self.itr_vec = [1000 for x in range(self.num_scale)]        

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")        

        self.muRGB = torch.from_numpy(np.asarray([[0.485],[0.456],[0.406]]).astype('float32')).to(self.device)
        self.stdRGB = torch.from_numpy(np.asarray([[0.229],[0.224],[0.225]]).astype('float32')).to(self.device)

        self.ref_model = network.Net(params.extracted_layers).to(self.device)
        self.stylize = network.Stats(params.stat_type).to(self.device)
        self.loss_model = network.LossModel().to(self.device)       
        self.loss_tv = network.LossTV().to(self.device)

        self._set_parameter_requires_grad(self.ref_model)

        self.out_dir = params.out_dir
        if not os.path.exists(params.out_dir):
            os.makedirs(params.out_dir)
        if self.log_tensorboard:
            self.tensorboard_dir = params.tensorboard_dir
            if not os.path.exists(params.tensorboard_dir):
                os.makedirs(params.tensorboard_dir)

    def _set_parameter_requires_grad(self, model):
        '''
        Turn off gradients
        '''

        for param in model.parameters():
            param.requires_grad = False

    def _get_ref_data(self, data_loader, itr_gen):
        '''
        Select a random reference image file from each precursor
        '''

        data_list = []; name_list = []

        for di, data in enumerate(data_loader):            
            #random.seed(di+2*itr_gen)
            ridx = 0            
            if len(data) > 1:
                ridx = random.randint(0,len(data)-1)        
            curr_data = data[ridx]
            print("# of files:", len(data), ' - Index selection: ', ridx)
        
            data_list.append(curr_data['img'])
            name_list.append(curr_data['img_name'])            
        return data_list, name_list

    def _load_refs(self, data_list, scale):
        '''
        Load statistics of each selected reference images
        '''

        ref_img_list = []; ref_stats_list = []; ref_ftrs_list = []
        for curr_data in data_list:            
            inp = torch.unsqueeze(curr_data.to(self.device),0) 
            inp = F.interpolate(inp, scale_factor=scale, mode='bilinear')                 
            ftrs_dict = self.ref_model(inp)
            stats_list, ftrs_list = self.stylize(ftrs_dict)

            ref_img_list.append(inp)
            ref_stats_list.append(stats_list)
            ref_ftrs_list.append(ftrs_list)

        return ref_img_list, ref_stats_list, ref_ftrs_list

    def _compute_weighted_stats(self,w, stats_list, weighted_stats_list):
        '''
        Compute weighted statistics of reference images
        '''

        for wg, g in zip(weighted_stats_list, stats_list):
            wg += w*g

        return weighted_stats_list

    def optimize(self, data_loader, opt_type, gen_w=256, gen_h=256):
                
        with torch.set_grad_enabled(True):
            for itr_gen in range(self.num_gen):
                self.logger.info("Load reference images for generated image #{:}".format(itr_gen+self.start_num))
                ## Load refs info ##
                curr_ref_data_list, curr_ref_name_list = self._get_ref_data(data_loader, itr_gen)                
                for rn, rname in enumerate(curr_ref_name_list):
                    self.logger.info("Ref#{:}: {:}".format(rn,rname))
                
                ## generate random weights ##
                #np.random.seed(1)
                if len(self.w_precursors) == 0:
                    weights = np.random.dirichlet(np.ones((len(self.precursors),)), size=(1,)) 
                    weights = np.squeeze( (weights * 100 + 0.5).astype('int') / 100.0 )
                else:
                    weights = np.asarray(self.w_precursors)                
                                
                ## set output name ##
                curr_out_name = '%s/gen' %self.out_dir
                for ni, n in enumerate(self.precursors):
                    curr_out_name += '_%d_%s' %(int(weights[ni]*100),n)                    
                curr_out_name += '_%d.tif' %(itr_gen + self.start_num)
                self.logger.info("Generated Image Name: {:} of size {:}".format(curr_out_name,(gen_w,gen_h)))

                ## Setup Tensorboard for curr image ##
                if self.log_tensorboard:
                    curr_save_model = curr_out_name.split('/')[-1].split('.')[0]
                    self.curr_writer = SummaryWriter('%s/%s' %(self.tensorboard_dir, curr_save_model))
                                                                                                                                                    
                start_time = time.time()
                for itr_scale in range(self.num_scale):
                    self.logger.info("Optimize at {:} scale of the generated image!".format(2**(itr_scale+1-self.num_scale)))

                    ## Initialize gen_image ##                               
                    if itr_scale == 0:
                        #np.random.seed(10)
                        random_noise = np.random.uniform(low=0.0, high=1.0, \
                                            size=(1,3,gen_h//(2**(self.num_scale-itr_scale-1)),gen_w//(2**(self.num_scale-itr_scale-1))))             
                        random_tensor = torch.from_numpy(random_noise).type(torch.cuda.FloatTensor) 
                    else:                        
                        random_tensor = F.interpolate(gen_img.detach(), scale_factor=2, mode='bilinear')  

                    gen_img = Variable(random_tensor, requires_grad=True) 
                    
                    ## Check if the gradient is None in gen_img ##
                    if gen_img.grad != None:
                        print('Gradient of generated image is not None'); return                          
                    
                    curr_num_itr = self.itr_vec[itr_scale]
                    curr_ref_img_list, curr_ref_stat_list, curr_ref_ftrs_list = \
                                                    self._load_refs(curr_ref_data_list, 2**(itr_scale+1-self.num_scale))
                    
                    ## Compute weighted ref statistics ##
                    ref_weighted_stat_list = []
                    [ref_weighted_stat_list.append(weights[0]*g) for g in curr_ref_stat_list[0]]
                
                    for gi, g_list in enumerate(curr_ref_stat_list[1:]):
                        ref_weighted_stat_list = self._compute_weighted_stats(weights[gi+1],g_list,ref_weighted_stat_list)

                    ## Check the weighted statistics matrices ##
                    for l in range(len(curr_ref_stat_list[0])):
                        sumg = 0.0; inds = (10,10); l = 0
                        for gi, g_list in enumerate(curr_ref_stat_list):
                            sumg += (weights[gi]* g_list[l][inds])
                        
                        if sumg != ref_weighted_stat_list[l][inds]:
                            print(sumg, ref_weighted_stat_list[l][inds])
                            print('Wrong aggregation!')
                            return
                                        
                    mu = torch.unsqueeze(torch.unsqueeze(self.muRGB,0),3).repeat(1,1,gen_img.size(2),gen_img.size(3))
                    std = torch.unsqueeze(torch.unsqueeze(self.stdRGB,0),3).repeat(1,1,gen_img.size(2),gen_img.size(3))
                    ## Start optimize ##                                                
                    if opt_type == 'Adam':   
                        self.optimizer = torch.optim.Adam([gen_img], lr=self.lr)                     
                        self._Adam_opt(gen_img, mu, std, ref_weighted_stat_list, curr_out_name, curr_num_itr)
                    else:                        
                        self.optimizer = torch.optim.LBFGS([gen_img], lr=self.lr, max_iter=self.num_itr+1)   
                        self._LBFGS_opt(gen_img, mu, std, ref_weighted_stat_list, curr_out_name, curr_num_itr)

                    ## Display refs and gen images on the same canvas ##                                
                    if self.display_ref_gen:
                        save_real_gen(curr_ref_img_list, gen_img, curr_out_name, mu, std, self.transf_type)                      

                if self.log_tensorboard:
                    self.curr_writer.close()
                self.logger.info('Total time = {:.5f}s\n'.format(time.time()-start_time))
                print('Finish generate image #%d\n' %(itr_gen+1))
                self.optimizer.zero_grad()


    def _Adam_opt(self, gen_img, mu, std, ref_weighted_stat_list, curr_out_name, num_itr):
        '''
        Optimize with Adam solver
        '''

        min_loss = 1e10                         
                
        since = time.time()
        for itr in range(num_itr+1): 
            self.optimizer.zero_grad()                                
            loss, loss_stat, loss_tv = self._compute_loss(gen_img, ref_weighted_stat_list, mu, std)                
            loss.backward()                                             
            self.optimizer.step()   
            if self.log_tensorboard:         
                self.curr_writer.add_scalar('Loss', np.log(loss.item()), itr)   
                self.curr_writer.add_scalar('Loss_Stat', np.log(loss_stat.item()), itr)
                self.curr_writer.add_scalar('Loss_TV', np.log(loss_tv.item()), itr)   
                     
            if itr != 0 and (itr % (num_itr//np.minimum(10,num_itr)) == 0):
                time_elapsed = time.time() - since                                                                           
                self.logger.info('Itr# {:} -> Loss = {:.5f} in {:.3f}s'\
                        .format(itr,loss.item(),time_elapsed))  
                since = time.time() 
                
                min_loss = self._save_img_checkpoint(gen_img, min_loss, loss.item(), itr, curr_out_name)                                           
        

    def _LBFGS_opt(self, gen_img, mu, std, ref_weighted_stat_list, curr_out_name, num_itr):
        '''
        Optimize with LBFGS solver
        '''

        min_loss = 1e10  
                                
        self.itr = 0; 
        self.min_loss = 1e10
        self.since = time.time()
        self.curr_num_itr = num_itr
        def closure():  
            self.optimizer.zero_grad()         
            loss, loss_stat, loss_tv = self._compute_loss(gen_img, ref_weighted_stat_list, mu, std)
            loss.backward() 

            if self.log_tensorboard:
                self.curr_writer.add_scalar('Loss', np.log(loss.item()), self.itr)   
                self.curr_writer.add_scalar('Loss_Stat', np.log(loss_stat.item()), self.itr)
                self.curr_writer.add_scalar('Loss_TV', np.log(loss_tv.item()), self.itr) 

            if self.itr != 0 and (self.itr % (self.curr_num_itr//np.minimum(10,self.curr_num_itr)) == 0):
                time_elapsed = time.time() - self.since                                                                           
                self.logger.info('Itr# {:} -> Loss = {:.5f} in {:.3f}s'\
                    .format(self.itr,loss.item(),time_elapsed))  
                self.since = time.time() 
                
                self.min_loss = self._save_img_checkpoint(gen_img, self.min_loss, loss.item(), self.itr, curr_out_name)            
        
            self.itr+=1
            return loss

        self.optimizer.step(closure)


    def _compute_loss(self, gen_img, ref_weighted_stat_list, mu, std):
        '''
        '''

        if self.transf_type == 0:
            out_tensor = (gen_img - mu) / std
            gen_ftrs_dict = self.ref_model(out_tensor)
        elif self.transf_type == 1:
            gen_img_act = torch.sigmoid(gen_img)                
            out_tensor = (gen_img_act - mu ) / std                    
            gen_ftrs_dict = self.ref_model(out_tensor)
        elif self.transf_type == 2:
            gen_ftrs_dict = self.ref_model(gen_img)

        gen_stat, gen_ftrs = self.stylize(gen_ftrs_dict) 
                               
        loss_stats = self.loss_model(gen_stat,  ref_weighted_stat_list)     
        loss_tv = self.loss_tv(gen_img)
        loss = self.w_stat * loss_stats + self.w_tv * loss_tv
        return loss, loss_stats, loss_tv


    def _save_img_checkpoint(self, gen_img, min_loss, curr_loss, itr, curr_out_name):
        '''
        Save generated image
        '''

        if self.transf_type == 0 or self.transf_type == 2:                            
            viz_img = (gen_img).detach().cpu().numpy()
            viz_grid = vutils.make_grid(torch.clamp((gen_img).detach(),0.0,1.0), nrow=1)
        elif self.transf_type == 1:
            gen_img_act = torch.sigmoid(gen_img.detach())
            viz_img = gen_img_act.cpu().numpy()
            viz_grid = vutils.make_grid(gen_img_act, nrow=1) 

        if min_loss > curr_loss:            
            min_loss = curr_loss                                
            save_img(viz_img,curr_out_name,isGray=True)     
            if self.log_tensorboard:                 
                self.curr_writer.add_image('Gen_Image', viz_grid, itr)

        return min_loss

        
            




 
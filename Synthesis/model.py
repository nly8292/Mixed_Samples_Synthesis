import tensorflow as tf
import numpy as np
import random
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import time
import scipy
import glob
import vgg16
import logging
from util import *

class Model(object):

    def __init__(self, params):
        
        self.num_groups = len(params.ref_dir)

        ##### Load files in each reference group #####
        self.ref_files_lists = []; self.ref_seg_lists = []

        for x in params.ref_dir:
            curr_group = '%s/%s/*' %(params.rdir, x)
            curr_files_list = glob.glob(curr_group)
            
            if params.gen_type == 'constraint':
                seg_files_list = []
                [seg_files_list.append(val) for val in curr_files_list if params.seg_ext in val]
                self.ref_seg_lists.append(seg_files_list)
                curr_files_list = list( set(curr_files_list) - set(seg_files_list) )

            self.ref_files_lists.append(curr_files_list)

        self.gen_seg_list = glob.glob(params.gen_seg_dir + '/*')
        
        self.params = params

    def get_ftrs(self, img_array):
        tf.reset_default_graph()

        vgg = vgg16.Vgg16(self.params.pretrained_file)
        vgg.build(img_array)
        content_layers_list = \
            dict({0: vgg.conv1_1, 1: vgg.conv1_2, 2: vgg.pool1, 3: vgg.conv2_1, 4: vgg.conv2_2, 5: vgg.pool2, \
            6: vgg.conv3_1, 7: vgg.conv3_2, 8: vgg.conv3_3, 9: vgg.pool3, 10: vgg.conv4_1, 11: vgg.conv4_2, 12: vgg.conv4_3, \
            13: vgg.pool4, 14: vgg.conv5_1, 15: vgg.conv5_2, 16: vgg.conv5_3, 17: vgg.pool5 })

        img_layer_outputs = dict()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())        
            for i in range(len(content_layers_list)):
                img_layer_outputs[i] = sess.run(content_layers_list[i])
            
        
        return img_layer_outputs

    def load_ref_unconstraint(self, ref_img_list, omega_vec):

        scale_ref_img = []; scale_ref_gram = []; scale_ref_ftrs = []; scale_gen_mask = []           

        for scale in range(self.params.num_scale):          
            
            ref_gram_list = {}; ref_ftrs_list = {}          
            
            for ref_num, ref_img_name in enumerate(ref_img_list):
                ##### Load reference image #####
                curr_img = load_img(ref_img_name, \
                            self.params.gen_w//(2**(self.params.num_scale-1-scale)), \
                            self.params.gen_h//(2**(self.params.num_scale-1-scale)), \
                            self.params.gen_w, self.params.gen_h)
                
                scale_ref_img.append(curr_img)
                ##### Load feature maps #####
                curr_img_ftrs = self.get_ftrs(curr_img)
            
                for i in range(len(self.params.layers)):                                        
                
                    curr_ftrs = np.squeeze( curr_img_ftrs[self.params.layers[i][0]], 0 )
                    ##### Load ref masks #####                    
                    curr_mask = np.sqrt(omega_vec[ref_num]) * np.ones((curr_ftrs.shape[0],curr_ftrs.shape[1]))                                                            
                    curr_mask = np.expand_dims(curr_mask,2); curr_mask = np.tile(curr_mask,(1,1,curr_ftrs.shape[2]))                                                        
                        
                    ##### Reshape ftrs #####
                    curr_ftrs = np.reshape( curr_ftrs, newshape=(curr_ftrs.shape[0]*curr_ftrs.shape[1],curr_ftrs.shape[2]))
                    curr_mask = np.reshape( curr_mask, newshape=(curr_mask.shape[0]*curr_mask.shape[1],curr_mask.shape[2])) 
                    
                    ##### Compute masked out ftrs and gram matrix #####
                    curr_ftrs = curr_ftrs * curr_mask                    
                    deno = curr_mask.shape[0]                    

                    curr_gram = np.matmul(curr_ftrs.T, curr_ftrs)
                    curr_gram /= deno   
                    
                    if ref_num == 0:
                        ref_gram_list['layer%d'%self.params.layers[i][0]] = [curr_gram]
                        ref_ftrs_list['layer%d'%self.params.layers[i][0]] = [curr_ftrs]                        
                    else:
                        ref_gram_list['layer%d'%self.params.layers[i][0]].append(curr_gram)
                        ref_ftrs_list['layer%d'%self.params.layers[i][0]].append(curr_ftrs)
                                            
            scale_ref_gram.append(ref_gram_list)    
            scale_ref_ftrs.append(ref_ftrs_list) 
            
        return scale_ref_img, scale_ref_ftrs, scale_ref_gram, scale_gen_mask


    def load_ref_constraint(self, ref_img_list, ref_seg_list, gen_seg):

        scale_ref_img = []; scale_ref_gram = []; scale_ref_ftrs = []; scale_gen_mask = []
           
        for scale in range(self.params.num_scale):          
            
            ref_gram_list = {}; ref_ftrs_list = {}; ref_gen_mask = {}           
            
            for ref_num, (ref_img_name, seg_img_name) in enumerate(zip(ref_img_list, ref_seg_list)):
                ##### Load reference image #####
                curr_img = load_img(ref_img_name, \
                            self.params.gen_w//(2**(self.params.num_scale-1-scale)), \
                            self.params.gen_h//(2**(self.params.num_scale-1-scale)), \
                            self.params.gen_w, self.params.gen_h)
                
                scale_ref_img.append(curr_img)
                ##### Load feature maps #####
                curr_img_ftrs = self.get_ftrs(curr_img)
                            
                curr_seg_img = load_img(seg_img_name, \
                            self.params.gen_w//(2**(self.params.num_scale-1-scale)), \
                            self.params.gen_h//(2**(self.params.num_scale-1-scale)), \
                            self.params.gen_w, self.params.gen_h)

                curr_gen_seg = load_img(gen_seg, \
                            self.params.gen_w//(2**(self.params.num_scale-1-scale)), \
                            self.params.gen_h//(2**(self.params.num_scale-1-scale)), \
                            self.params.gen_w, self.params.gen_h)

                if 'GLAS' in seg_img_name and ref_num == 1:
                    curr_seg_img = 1 - curr_seg_img                
                elif 'GLAS' not in seg_img_name:
                    curr_gen_seg[np.where(curr_gen_seg != ref_num)] = 100
                    curr_gen_seg[np.where(curr_gen_seg == ref_num)] = 1
                    curr_gen_seg[np.where(curr_gen_seg == 100)] = 0

                plt.figure()
                print(ref_img_name.split('/')[-1][:-4])
                plt.imshow(np.squeeze(curr_seg_img)); plt.savefig('./%s_%d_%d.png' %(ref_img_name.split('/')[-1][:-4],ref_num,scale))

                for i in range(len(self.params.layers)):                                        
                
                    curr_ftrs = np.squeeze( curr_img_ftrs[self.params.layers[i][0]], 0 )

                    ##### Load ref masks #####                    
                    curr_mask = skimageRz(curr_seg_img[0,...,0],(curr_ftrs.shape[0],curr_ftrs.shape[1])).astype(np.float32)                
                    curr_mask = np.expand_dims(curr_mask,2); curr_mask = np.tile(curr_mask,(1,1,curr_ftrs.shape[2]))   

                    ##### Load gen mask #####  
                    if 'GLAS' in seg_img_name and ref_num == 1:      
                        curr_gmask = 1-ref_gen_mask['layer%d'%self.params.layers[i][0]][0]                                              
                    else:                                                   
                        curr_gmask = skimageRz(curr_gen_seg[0,...,0],(curr_ftrs.shape[0],curr_ftrs.shape[1])).astype(np.float32)  
                        #curr_gmask[np.where(curr_gmask > 0.0)] = 1.0                                      
                        curr_gmask = np.expand_dims(curr_gmask,2); curr_gmask = np.tile(curr_gmask,(1,1,curr_ftrs.shape[2]))
                        curr_gmask = np.reshape( curr_gmask, newshape=(curr_gmask.shape[0]*curr_gmask.shape[1],curr_gmask.shape[2]))
                    
                    tt = np.copy(curr_gmask[...,0])
                    tt = np.reshape( tt, newshape=(curr_ftrs.shape[0],curr_ftrs.shape[1]))
                    print(np.unique(curr_gen_seg),np.unique(tt))
                    plt.figure()
                    plt.imshow(tt); plt.savefig('./gen_%d_%d_%d.png' %(ref_num,scale,i))    

                    ##### Reshape ftrs #####
                    curr_ftrs = np.reshape( curr_ftrs, newshape=(curr_ftrs.shape[0]*curr_ftrs.shape[1],curr_ftrs.shape[2]))
                    curr_mask = np.reshape( curr_mask, newshape=(curr_mask.shape[0]*curr_mask.shape[1],curr_mask.shape[2])) 
                    
                    ##### Compute masked out ftrs and gram matrix #####
                    curr_ftrs = curr_ftrs * curr_mask
                    deno = 2*len(np.where(curr_mask[:,0] > 0)[0])                                      

                    curr_gram = np.matmul(curr_ftrs.T, curr_ftrs)
                    curr_gram /= deno   
                    
                    if ref_num == 0:
                        ref_gram_list['layer%d'%self.params.layers[i][0]] = [curr_gram]
                        ref_ftrs_list['layer%d'%self.params.layers[i][0]] = [curr_ftrs]
                        ref_gen_mask['layer%d'%self.params.layers[i][0]] = [curr_gmask]
                    else:
                        ref_gram_list['layer%d'%self.params.layers[i][0]].append(curr_gram)
                        ref_ftrs_list['layer%d'%self.params.layers[i][0]].append(curr_ftrs)
                        ref_gen_mask['layer%d'%self.params.layers[i][0]].append(curr_gmask)

                    
            scale_ref_gram.append(ref_gram_list)    
            scale_ref_ftrs.append(ref_ftrs_list)
            scale_gen_mask.append(ref_gen_mask) 
            

        return scale_ref_img, scale_ref_ftrs, scale_ref_gram, scale_gen_mask


    def gen_model(self):

        for numItr_gen in range(self.params.num_gen):
            logging.info("Generate sample #{:d}".format(numItr_gen))
            ##### Select reference images #####   
            curr_ref_image_names = []
            for c, x in enumerate(self.ref_files_lists):                
                randnum = random.randint(0,len(x)-1)   
                while x[randnum] in curr_ref_image_names:
                    randnum = random.randint(0,len(x)-1)            
                curr_ref_image_names.append(x[randnum]) 

                logging.info("Ref Image #{:}: {:}".format(c,x[randnum]))
            
            curr_ref_image_names = [self.params.rdir + '/fg/train_2.bmp', self.params.rdir + '/bg/train_48.bmp']
            if self.params.gen_type == 'unconstraint':
                curr_w = np.random.dirichlet( np.ones((self.num_groups,)), size=(1,))  
                #curr_w = np.asarray([1.0, 0.0])
                curr_w = np.round(curr_w*100.0)/100.0; curr_w = curr_w.reshape(self.num_groups,) 

                ##### Load reference data #####
                scale_ref_img, scale_ref_ftrs, scale_ref_gram, scale_gen_mask = \
                        self.load_ref_unconstraint(curr_ref_image_names, curr_w)  

                curr_gen_seg = ''                      
            else:                
                ##### Select corresponding mask ###############
                curr_ref_seg_names = []
                for ref_img_name in curr_ref_image_names:
                    curr_ref_seg_names.append(ref_img_name[:-4] + '_' + self.params.seg_ext) 

                    logging.info("Ref Seg Image #{:}: {:}".format(c,curr_ref_seg_names[-1]))

                if 'GLAS' in self.params.rdir:
                    curr_gen_seg = curr_ref_seg_names[0]
                else:
                    randnum = random.randint(0,len(gen_seg_list)-1)  
                    curr_gen_seg = self.gen_seg_list[randnum]  

                logging.info("Generated Image Mask: {:}".format(curr_gen_seg))          

                ##### Load reference data #####
                scale_ref_img, scale_ref_ftrs, scale_ref_gram, scale_gen_mask = \
                        self.load_ref_constraint(curr_ref_image_names, curr_ref_seg_names, curr_gen_seg)

            ####### Create generated image file name #######################################
            output_filename = self.params.output_dir + 'gen_' 
            for c, x in enumerate(curr_ref_image_names):
                if self.params.seg_ext == '':
                    output_filename += '%d%s_' %(int(curr_w[c]*100),self.params.ref_dir[c])
                else:
                    temp_name = x.split('/')[-1][:-4]
                    output_filename += '%s_' %temp_name
            output_filename += '%d.jpg' %(numItr_gen+self.params.num_start)
            logging.info("Out File: {:}".format(output_filename))
                      
            print(len(scale_ref_img), len(scale_ref_gram))           
            ##### Optimize #####
            self.optimize(scale_ref_img, scale_ref_ftrs, scale_ref_gram, scale_gen_mask,\
                            output_filename, curr_ref_image_names, curr_gen_seg)


    def optimize(self, ref_img, ref_ftrs_list, ref_gram_list, gen_mask_list,\
                output_filename, ref_image_names, gen_seg_name):

        ##### Build ftrs for noise input #####
        total_start_time = time.time()
        loss_vec = []       
        vec = [100,1000,int(0.75*self.params.num_itr)]

        for scale in range(self.params.num_scale):  
                               
            scale_num_itr = vec[scale]       
            tf.reset_default_graph()
            vgg = vgg16.Vgg16(self.params.pretrained_file)

            if scale == 0:              
                random_ = tf.random_uniform(shape=ref_img[scale].shape, minval=0, maxval=1.00)           
                inp_rand = tf.Variable(initial_value=random_, name='input_noise_%d' %scale, dtype=tf.float32)               
            else:                               
                x = np.squeeze(random_)#;x = (x - np.amin(x)) / (np.amax(x) - np.amin(x))    
                x *= 255; x = np.clip(x, 0, 255).astype('uint8')

                img = Image.fromarray(x, mode='RGB')                                        
                img = img.resize((\
                    self.params.gen_w//(2**(self.params.num_scale-1-scale)), \
                    self.params.gen_h//(2**(self.params.num_scale-1-scale))), resample=Image.BILINEAR)              
                img_rand = np.expand_dims(np.array(img, dtype=np.float32),0) / 255.0                    
                inp_rand = tf.Variable(initial_value=img_rand, name='input_noise_%d' %scale, dtype=tf.float32)              
            
            input_noise = tf.nn.sigmoid(inp_rand)
            vgg.build(input_noise)

            noise_layers_list = \
                dict({0: vgg.conv1_1, 1: vgg.conv1_2, 2: vgg.pool1, 3: vgg.conv2_1, 4: vgg.conv2_2, 5: vgg.pool2, \
                    6: vgg.conv3_1, 7: vgg.conv3_2, 8: vgg.conv3_3, 9: vgg.pool3, 10: vgg.conv4_1, 11: vgg.conv4_2, \
                    12: vgg.conv4_3, 13: vgg.pool4, 14: vgg.conv5_1, 15: vgg.conv5_2, 16: vgg.conv5_3, 17: vgg.pool5 })

            if self.params.gen_type == 'unconstraint':
                loss, loss_content, loss_texture, loss_tv = \
                        self.loss_function_unconstraint(input_noise, noise_layers_list, \
                                        ref_img[scale], ref_ftrs_list[scale],\
                                        ref_gram_list[scale])                    
            else:
                loss, loss_content, loss_texture, loss_tv = \
                        self.loss_function_constraint(input_noise, noise_layers_list, \
                                        ref_img[scale], ref_ftrs_list[scale],\
                                        ref_gram_list[scale], gen_mask_list[scale])                               

            optimizer = tf.train.AdamOptimizer(learning_rate=self.params.lr).minimize(loss)  
        
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True

            with tf.Session(config=config) as sess:
                sess.run(tf.global_variables_initializer())             
            
                start_time = time.time()
                
                if self.params.num_scale-1-scale == 0:
                    loss_vec_scale = []; lossc_vec_scale = []; losst_vec_scale = []

                for it in range(scale_num_itr+1):            
                    _, curr_loss, curr_lc, curr_lt, curr_ltv, updated_noise, updated_rand = \
                                sess.run([optimizer, loss, loss_content, loss_texture, loss_tv, input_noise, inp_rand])

                    loss_vec.append(curr_loss)
                    if self.params.num_scale-1-scale == 0:
                        loss_vec_scale.append(curr_loss)
                        lossc_vec_scale.append(curr_lc)
                        losst_vec_scale.append(curr_lt)
                    
                    if it % (scale_num_itr//np.minimum(5,scale_num_itr)) == 0:
                        run_time = time.time() - start_time
                        logging.info(\
                        "Epoch: {}/{} Loss: {:.4f} Loss_Content: {:.4f} Loss_Texture: {:.4f} Loss_TV {:.4f} Run time: {:.4f}"\
                        .format(it, scale_num_itr, curr_loss, curr_lc, curr_lt, curr_ltv, run_time))                
                        
                        start_time = time.time()

                        if it == 0:
                            min_loss = curr_loss
                        elif min_loss > curr_loss and it != 0:
                            min_loss = curr_loss
                            ##### Save current generated image #####
                            if self.params.num_scale-1-scale == 0:                              
                                post_process(updated_noise, ref_image_names, gen_seg_name, output_filename)#[:-4] + '_scale%d_%d.pn
                                #post_process(updated_rand, ref_image_names, gen_seg_name, output_filename[:-4] + '_scaleR%d_%d.png'%(scale,it))
                            else:                               
                                random_ = np.copy(updated_noise)
                                #post_process(updated_noise, ref_image_names, gen_seg_name, output_filename[:-4] + '_scale%d_%d.png'%(scale,it))                                

            if self.params.num_scale-1-scale == 0:
                plt.figure()
                plt.plot(np.log(np.asarray(loss_vec_scale)),'r')
                plt.plot(np.log(np.asarray(lossc_vec_scale)),'k') 
                plt.plot(np.log(np.asarray(losst_vec_scale)),'g') 
                plt.legend(['Loss', 'Loss_Content', 'Loss_Texture'])
                plt.savefig('./loss_%s_scale%d.png' %(output_filename.split('/')[-1][:-4],scale))

        plt.figure(); plt.plot(np.log(np.asarray(loss_vec)),'r'); 
        plt.savefig('./loss_%s.png' %(output_filename.split('/')[-1][:-4]))

        logging.info("Total opt time: {:.4f}".format(time.time()-total_start_time))


    def loss_function_unconstraint(self, input_noise, noise_layers, \
                ref_img, ref_ftrs_list, ref_gram_list):

        ###### Loss without structrual constraint #########################################
        loss_texture = tf.constant(0, dtype=tf.float32, name="Loss_Texture")
        loss_content = tf.constant(0, dtype=tf.float32, name="Loss_Content")
        for i in range(len(self.params.layers)): 

            noise_filters = tf.squeeze(noise_layers[self.params.layers[i][0]], 0)           
            noise_filters = tf.reshape(noise_filters, shape=(noise_filters.shape[0] * noise_filters.shape[1], \
                                        noise_filters.shape[2]))

            deno = tf.convert_to_tensor(ref_ftrs_list['layer%d'%self.params.layers[i][0]][0].shape[0], dtype=tf.float32) 
            gram_noise = tf.matmul(tf.transpose(noise_filters), noise_filters)
            gram_noise /= deno

            for ii, ref_gram in enumerate(ref_gram_list['layer%d'%self.params.layers[i][0]]):
                if ii == 0:
                    curr_ref_gram = ref_gram
                else:
                    curr_ref_gram += ref_gram
                                    
            loss_texture += self.params.layers[i][1] * \
                    tf.reduce_sum(tf.square(tf.subtract(tf.cast(curr_ref_gram,tf.float32), gram_noise)))        

        loss_tv = self.params.gamma*tf.image.total_variation(input_noise)[0] / \
                    tf.cast(input_noise.shape[1]*input_noise.shape[2], tf.float32)
        
        loss_texture *= self.params.alpha        

        loss = loss_texture + loss_tv

        return loss, loss_content, loss_texture, loss_tv

    def loss_function_constraint(self, input_noise, noise_layers, \
                ref_img, ref_ftrs_list, ref_gram_list,  gen_mask_list):

        ######### Loss with structural constraint ########################
        loss_texture = tf.constant(0, dtype=tf.float32, name="Loss_Texture")
        loss_content = tf.constant(0, dtype=tf.float32, name="Loss_Content")
        for i in range(len(self.params.layers)):            
            noise_filters = tf.squeeze(noise_layers[self.params.layers[i][0]], 0)           
            noise_filters = tf.reshape(noise_filters, shape=(noise_filters.shape[0] * noise_filters.shape[1], \
                                        noise_filters.shape[2]))
                                      
            loss_texture_l = tf.constant(0, dtype=tf.float32, name="Loss")
            for ii, (ref_gram, ref_ftrs, gen_mask) \
                in enumerate(zip(ref_gram_list['layer%d'%self.params.layers[i][0]], \
                    ref_ftrs_list['layer%d'%self.params.layers[i][0]], \
                    gen_mask_list['layer%d'%self.params.layers[i][0]])):
                
                tensor_mask = tf.convert_to_tensor(gen_mask, dtype=tf.float32)
                deno = 2*tf.convert_to_tensor(len(np.where(gen_mask[:,0] > 0)[0]), dtype=tf.float32)                    
                
                curr_noise = noise_filters*tensor_mask
                gram_noise = tf.matmul(tf.transpose(curr_noise), curr_noise)
                gram_noise /= deno

                if i == 3:                    
                    denominator = (tf.convert_to_tensor(gen_mask.shape[0]*gen_mask.shape[1], dtype=tf.float32))
                    
                    if 'GLAS' not in self.ref_seg_lists[0][0] or ii == 0 :
                        print(ii,gen_mask.shape[0],gen_mask.shape[1])
                        loss_content += ((tf.reduce_sum(tf.square(tf.subtract(ref_ftrs, curr_noise))) )\
                                        / tf.cast(denominator, tf.float32))
                    
                loss_texture_l += tf.reduce_sum(tf.square(tf.subtract(ref_gram, gram_noise)))

            loss_texture += self.params.layers[i][1] * loss_texture_l

            

        loss_tv = self.params.gamma*tf.image.total_variation(input_noise)[0] / \
                    tf.cast(input_noise.shape[1]*input_noise.shape[2], tf.float32)
        
        loss_texture *= self.params.alpha
        loss_content *= self.params.beta

        loss = loss_content + loss_texture + loss_tv

        return loss, loss_content, loss_texture, loss_tv

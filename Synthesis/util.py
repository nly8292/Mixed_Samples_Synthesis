from skimage.transform import match_histograms
from skimage import exposure
from skimage.transform import resize as skimageRz
from skimage.feature import canny
from PIL import Image
import cv2
import os
import numpy as np
import logging


def load_img(image_path, img_w, img_h, gen_w, gen_h):
        
    if os.path.isfile(image_path):
        img = Image.open(image_path)                
                                              
        if img_w != gen_w and img_h != gen_h:
            img = img.resize((img_w,img_h), resample=Image.BILINEAR)

        img_array = np.array(img, dtype=np.float32)            
        img_array = img_array / 255.0

        if len(img_array.shape) < 3:                                
            img_array[np.where(img_array > 0.0)] = 1.0                
            img_array = np.expand_dims(img_array, 2)
            img_array = np.tile(img_array, (1,1,3))

        img_array = np.expand_dims(img_array, 0)
        
        return img_array
    else:
        logging.info("No image found in given location.")


def post_process(output, org_path, gen_mask_path, output_filename, save_file=True, isHistMatch=False, isPlotHist=False):
    
    ##### Shift pixel values #####
    x = np.squeeze(output)      
    #x = (x - np.amin(x)) / (np.amax(x) - np.amin(x))    
    x *= 255
    #x = np.mean(x,axis=2,keepdims=True)
    #x = np.tile(x,(1,1,3))
    x = np.clip(x, 0, 255).astype('uint8')

    img = Image.fromarray(x, mode='RGB')
    if save_file:
        img.save(output_filename)
        
    ######## Overlay Seg #################
    if len(gen_mask_path) > 0:
        mask = ( Image.open(gen_mask_path) ) 
        
        mask = np.asarray(mask).astype(np.float32)          
        if len(mask.shape) > 2:
            edges = canny(mask[...,0], sigma=3) 
        else:
            edges = canny(mask, sigma=3)
        mask_ov = np.copy(x)
        
        r = mask_ov[...,0]; r[np.where(edges > 0)] = 255
        g = mask_ov[...,1]; g[np.where(edges > 0)] = 0
        b = mask_ov[...,2]; b[np.where(edges > 0)] = 0
        mask_ov[...,0] = r; mask_ov[...,1] = g; mask_ov[...,2] = b      

        mask_img = Image.fromarray(mask_ov.astype('uint8'), mode='RGB') 
        mask_img.save(output_filename[:-4] + '_mask.jpg')

    ####### Hist matching ########
    if isHistMatch:
        org_img = Image.open(org_path[np.minimum(2,len(org_path)-1)])
        curr_real = np.array(org_img)    
        x_match = match_histograms(x, curr_real, multichannel=True)    
        match_img = Image.fromarray(x_match, mode='RGB')
        match_img.save(output_filename[:-4] + 'histmatch' + output_filename[-4:])    
            
    ##### Plot histogram #####
    if isPlotHist:
        cl = ['r', 'g', 'b']
        sym = ['-^','-*','-s','-+']

        orgimg_list = []
        for f in org_path:
            img = np.asarray(Image.open(f))
            orgimg_list.append( img )

        fg,ax = plt.subplots(3,1)
        for i in range(3):        
            histx = cv2.calcHist([x], [i], None, [256], [0,256])     
            ax[i].plot(histx/np.max(histx),cl[i])
            for c,y in enumerate([orgimg_list[0]]):  
                histy = cv2.calcHist([y], [i], None, [256], [0,256])           
                ax[i].plot(histy/np.max(histy),cl[i]+sym[c])    

        plt.tight_layout()
        plt.savefig('./hist_%s.png' %output_filename.split('/')[-1][:-4])
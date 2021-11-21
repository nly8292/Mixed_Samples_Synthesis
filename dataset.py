from __future__ import print_function, division, absolute_import, unicode_literals
from PIL import Image as PILImage
import numpy as np
import glob

from torch.utils.data import Dataset
from torchvision import transforms

MU_RGB = [0.485, 0.456, 0.406]
STD_RGB = [0.229, 0.224, 0.225]

class Mixtures(Dataset):

    def __init__(self, rdir, transform_type=0):        
        
        files = glob.glob(rdir + '/*')       
        
        self.im_list = files
        self.num_file = len(self.im_list)

        self.data_transforms = self._get_transforms(transform_type)
        

    def __len__(self):
        return len(self.im_list)

    def __getitem__(self, idx):
        
        img_name = self.im_list[idx]                  
        img = PILImage.open(img_name)  
                                                                             
        img = self._convert_img_mode(img)        
        img_ten = self.data_transforms(img)         
        
        return {'img':img_ten, 'img_name':img_name}


    def _convert_img_mode(self, img):
        '''
        Convert to RGB mode if necessary 
        '''

        if img.mode == 'RGB':
            img = np.array(img)
        elif img.mode == 'L' or img.mode == 'CMYK':
            img = img.convert('RGB')
            img = np.array(img)

        elif img.mode == 'I;16' or img.mode == 'I;16L' \
            or img.mode == 'I;16B' or img.mode == 'I;16N':

            vimg = img.convert('F')
            ## Check if the conversion change the range of values ##
            if np.array(img).max() != np.array(vimg).max() or \
                np.array(img).min() != np.array(vimg).min():
                print('Warning Incorrect convertion between modes!')
        
            img = img.convert('F')
        
            img = np.array(img)
            ## Convert to RGB ##
            if len(img.shape) < 3:                                            
                img = np.expand_dims(img, 2)
                img = np.tile(img, (1,1,3))
            ## Rescale ##
            if np.max(img) > 255:            
                img /= float(np.iinfo(np.uint16).max)
        else:
            raise Exception("Unsupport Image Format!!!")
        return img

    def _get_transforms(self, transform_type):
        '''
        Setup data augmentation process
        '''

        if transform_type < 2:
            data_transforms = transforms.Compose([\
                transforms.ToTensor(),
                transforms.Normalize(MU_RGB , STD_RGB)
                ])
        else:
            data_transforms = transforms.Compose([transforms.ToTensor()])

        return data_transforms

    def _get_randcoord(self,w,h,crop_size):
        x = np.random.randint(0,high=w-crop_size)
        while x+crop_size > w:
            x = np.random.randint(0,high=w-crop_size)
        y = np.random.randint(0,high=h-crop_size)
        while y+crop_size > h:
            y = np.random.randint(0,high=h-crop_size)

        return x, y

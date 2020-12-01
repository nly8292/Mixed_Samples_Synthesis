import numpy as np
import tensorflow as tf
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import logging
import datetime

import param as Params
import model

curr = datetime.datetime.now()
logging.basicConfig(filename='./%s_info.log'\
            %(str(curr.year)+'_'+str(curr.month)+'_'+str(curr.day)+'_'+str(curr.hour)+str(curr.minute)+str(curr.second))\
            ,level=logging.INFO, format='%(asctime)s %(message)s')
          
      
def Main():

    ##### Parse params #####
    params = Params.Params()
    params.parse_params()
    params.store_params()

    ##### Run the model #####
    syn_model = model.Model(params)
    syn_model.gen_model()


Main()
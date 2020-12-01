import logging
import argparse
import os

class Params():
    def __init__(self):

        self.parser = argparse.ArgumentParser() 

    def parse_params(self):
        self.parser.add_argument('-g', '--gpu', type=int, help = 'Assign GPU for the model')
        self.parser.add_argument('--num_start', type=int, default = 0, help = 'Start index of the generated image')    
        self.parser.add_argument('--num_itr', type=int, default = 100, help = 'Number of iterations')
        self.parser.add_argument('--num_gen', type=int, default = 1, help = 'Number of generated samples')
        self.parser.add_argument('--rdir', type=str, default = './', help = 'Parent directory of reference images')
        self.parser.add_argument('--ref_dir', type=str, nargs='*', default = [], help = 'Directory of reference classes')
        self.parser.add_argument('--gen_seg_dir', type=str, default = './', help = 'Directory of generated mask')
        self.parser.add_argument('--pretrained_file', type=str, default='', help = 'Pretraied file name')
        self.parser.add_argument('--gen_w', type=int, default = 224, help = 'Width of generated image')
        self.parser.add_argument('--gen_h', type=int, default = 224, help = 'Height of generated image')
        self.parser.add_argument('--lr', type=float, default = 1e-3, help = 'Learning rate')
        self.parser.add_argument('--alpha', type=float, default = 1e-3, help = 'Factor of Texture Loss')
        self.parser.add_argument('--beta', type=float, default = 1e-3, help = 'Factor of Content Loss')
        self.parser.add_argument('--gamma', type=float, default = 1e-3, help = 'Factor of TV Loss')
        self.parser.add_argument('--seg_ext', type=str, default = '', help = 'Mask extension name')
        self.parser.add_argument('--gen_type', type=str, default = '', help = 'Type of mixing')


        FLAGS, unparsed = self.parser.parse_known_args()
        self.num_start = FLAGS.num_start
        self.num_itr = FLAGS.num_itr
        self.num_gen = FLAGS.num_gen  

        self.alpha = FLAGS.alpha
        self.beta = FLAGS.beta
        self.gamma = FLAGS.gamma

        self.seg_ext = FLAGS.seg_ext
        self.gen_seg_dir = FLAGS.gen_seg_dir
        self.gen_type = FLAGS.gen_type

        gpu_req = FLAGS.gpu
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_req)

        ##### Layers use for feature maps extraction #####
        self.layers = [(0, 1), (3, 1), (6, 1), (10, 1), (14, 1)]
        #self.layers = [(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1), (8, 1), (9, 1), (10, 1), (11, 1), (12, 1), (13, 1)]

        self.rdir = FLAGS.rdir
        self.ref_dir = [item for item in FLAGS.ref_dir[0].split(',')]

        self.output_dir = './outputs/'
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.gen_w = FLAGS.gen_w
        self.gen_h = FLAGS.gen_h

        self.pretrained_file = FLAGS.pretrained_file

        self.lr = FLAGS.lr

        self.num_scale = 3


    def store_params(self):

        logging.info('----- Params -----')
        logging.info('num_start = {:}; num_itr = {:}; num_gen = {:}; lr = {:}'.\
                        format(self.num_start, self.num_itr, self.num_gen,self.lr))
        logging.info('Generated Image W = {:} - H = {:}'.format(self.gen_w,self.gen_h))
        logging.info('Layers: {:}'.format(self.layers))
        logging.info('Pretrained file: {:}\n'.format(self.pretrained_file))

        logging.info('Root dir for reference images: {:}'.format(self.rdir))
        logging.info('Dir of reference groups: {:}\n'.format(self.ref_dir))

        logging.info('Alpha = {:}; Beta = {:}; Gamma = {:}'.format(self.alpha,self.beta,self.gamma))

        logging.info('-------------------')
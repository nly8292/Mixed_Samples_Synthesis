# Mixed Samples Synthesis

This is an official implementation of a paper, "Synthesizing Mixed Image Samples with Neural Style Transfer for Data Augmentation".

## Requirements
Please install requirements by `pip install -r requirements.txt`

- Python 3.6+
- Tensorflow 1.14.0
- Pillow 6.1.0
- opencv-python 4.1.1.26
- scikit-image 0.15.0 

## Usage
### Pre-trained Models
The pre-trained weights of VGG16 for synthesizing images can be downloaded with this [link]. The weights need to be stored in `Synthesis/`.

### Dataset Directory Setup
The desired dataset has to be structured where the images of each class is the sub-folder within the main dataset folder e.g., `<data_root_dir\classA\>', `<data_root_dir\classB\>', etc... For the synthesizing constraints images, the segmentation masks have also to be structured in the similar way.

### Synthesize Images
To generate mixed sample image with no constraint, run `./run_unconstraint.sh` with the following modifications for your desired dataset:

* `-g`: GPU ID 
* `--num_start`: the index of the first generated image for this run
* `--num_itr`: the number of iterations to optimize each image
* `--rdir`: the root directory of dataset
* `--ref_dir`: the sub-directory of each class in the dataset
* `--pretrained_file`: the pre-trained weights file
* `--lr`: learning rate
* `--alpha`: weight of texture loss
* `--beta`: weight of content loss
* `--gamma`: weight of TV loss
* `--gen_w`: width of the generated image
* `--gen_h`: height of the generated image

To generate mixed sample image with constraints, run `./run_constraint.sh` with the following modifications for your desired dataset:

* `-g`: GPU ID 
* `--num_start`: the index of the first generated image for this run
* `--num_itr`: the number of iterations to optimize each image
* `--rdir`: the root directory of dataset
* `--ref_dir`: the sub-directory of each class in the dataset
* `--seg_ext`: the extension of segmentation mask
* `--gen_seg_dir`: the directory of segmentation masks
* `--pretrained_file`: the pre-trained weights file
* `--lr`: learning rate
* `--alpha`: weight of texture loss
* `--beta`: weight of content loss
* `--gamma`: weight of TV loss
* `--gen_w`: width of the generated image
* `--gen_h`: height of the generated image







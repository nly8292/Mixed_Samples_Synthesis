
# Mixed Sample Synthesis
This is the official implementation of mixed sample synthesis model in the paper ["dafafsda"]

## Setup

### Prerequisites
- Linux
- NVIDIA GPU + CUDA CuDNN (CPU mode may work without any modification, but untested)
- Python 3.6
- Install necessary libraries with pip
```
pip3 install -r requirements.txt
```

### Dataset
The dataset directory needs to be structured as follows:
```
${DATASET_ROOT_DIR}
|____DIR_PRECURSOR1
     |___image1
     |___image2
     |___image3
     ...
|____DIR_PRECURSOR2
     |___image1
     |___image2
     |___image3
     ...
|____DIR_PRECURSOR3
     |___image1
     |___image2
     |___image3
     ...
...                
```

## Synthesize Mixed Samples
Execute the script `run_script.sh` to generate mixed sample. Moreover, the following params inside `run_script.sh` can be modified to obtain desired behavior.
-`gpu_id` - GPU ID
-`num_scale` - The number of levels in the pyramid optimization
-`itr_vec` - The number of iterations at each scale
-`lr` - learning rate
-`start_num` - The starting index of the generated images
-`num_gen` - The number of synthetic images to be generated
-`precursors` - The precursors used for generating mixed samples
-`w_precursors` - The contribution of each precursor to the generated images
-`transf_type` - Data augmentation method `0:Normalize image input using ImageNet statistics | 1:Add sigmoid activation for generated on top of normalization of 0 | 2:No normalization`    
-`extracted_layers` - The layers from VGG16 to be used for computing statistics
-`w_stat` - Weight param for statistics loss
-`w_tv` - Weighted param for TV loss'    
-`gen_w` - Width of generated image
-`gen_h` - Height of generated image    
-`opt_type` - Type of optimizer `Adam (Default) | LBFGS`
-`stat_type` - Type of statistic to be computed for the extracted layers
-`root_dir` - Dataset root directory `${DATASET_ROOT_DIR}`    
-`out_dir` - Directory for storing generated images
-`tensorboard_dir` - Directory for storing tensorboard results
-`log_tensorboard` - Turn on tensorboard
-`display_ref_gen` - Displaying reference images and generated image on the same canvas


## Citation
If you use this code for your research, please cite our paper <a href="to_be_added">Determining the Composition of a Mixed Material with Synthetic Data</a>:

```
bibtex_to_be_added
```


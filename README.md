# Unet-Segmentation-Pytorch-Nest-of-Unets

[![forthebadge](https://forthebadge.com/images/badges/made-with-python.svg)](https://www.python.org/)

[![HitCount](http://hits.dwyl.io/bigmb/Unet-Segmentation-Pytorch-Nest-of-Unets.svg)](http://hits.dwyl.io/bigmb/Unet-Segmentation-Pytorch-Nest-of-Unets)
[![License: MIT](https://img.shields.io/badge/License-MIT-brightgreen.svg)](https://opensource.org/licenses/MIT)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/bigmb/Unet-Segmentation-Pytorch-Nest-of-Unets/graphs/commit-activity)
[![GitHub issues](https://img.shields.io/github/issues/Naereen/StrapDown.js.svg)](https://github.com/bigmb/Unet-Segmentation-Pytorch-Nest-of-Unets/issues)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/unet-a-nested-u-net-architecture-for-medical/semantic-segmentation-on-cityscapes-val)](https://paperswithcode.com/sota/semantic-segmentation-on-cityscapes-val?p=unet-a-nested-u-net-architecture-for-medical)

Implementation of different kinds of Unet Models for Image Segmentation

1) **UNet** - U-Net: Convolutional Networks for Biomedical Image Segmentation
https://arxiv.org/abs/1505.04597

2) **RCNN-UNet** - Recurrent Residual Convolutional Neural Network based on U-Net (R2U-Net) for Medical Image Segmentation
https://arxiv.org/abs/1802.06955

3) **Attention Unet** - Attention U-Net: Learning Where to Look for the Pancreas
https://arxiv.org/abs/1804.03999

4) **RCNN-Attention Unet** - Attention R2U-Net : Just integration of two recent advanced works (R2U-Net + Attention U-Net)
<!--LeeJun Implementation - https://github.com/LeeJunHyun/Image_Segmentation.git -->

5) **Nested UNet** - UNet++: A Nested U-Net Architecture for Medical Image Segmentation
https://arxiv.org/abs/1807.10165

With Layer Visualization

## 1. Getting Started

Clone the repo:

  ```bash
  git clone https://github.com/bigmb/Unet-Segmentation-Pytorch-Nest-of-Unets.git
  ```

## 2. Requirements

```
python>=3.6
torch>=0.4.0
torchvision
torchsummary
tensorboardx
natsort
numpy
pillow
scipy
scikit-image
sklearn
```
Install all dependent libraries:
  ```bash
  pip install -r requirements.txt
  ```
## 3. Run the file

Add all your folders to this line 106-113
```
t_data = '' # Input data
l_data = '' #Input Label
test_image = '' #Image to be predicted while training
test_label = '' #Label of the prediction Image
test_folderP = '' #Test folder Image
test_folderL = '' #Test folder Label for calculating the Dice score
 ```
 
  ## 4. Types of Unet
  
  **Unet**
  ![unet1](/images/unet1.png)
  
  **RCNN Unet**
  ![r2unet](/images/r2unet.png)
  
  
  **Attention Unet**
  ![att-unet](/images/att-unet.png)
  
  
  **Attention-RCNN Unet**
  ![att-r2u](/images/att-r2u.png)
  
  
  **Nested Unet**
  
  ![nested](/images/nested.jpg)

## 5. Visualization

To plot the loss , Visdom would be required. The code is already written, just uncomment the required part.
Gradient flow can be used too. Taken from (https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/10)

A model folder is created and all the data is stored inside that.
Last layer will be saved in the model folder. If any particular layer is required , mention it in the line 361.

**Layer Visulization**

![l2](/images/l2.png)

**Filter Visulization**

![filt1](/images/filt1.png)

**TensorboardX**
Still have to tweak some parameters to get visualization. Have messed up this trying to make pytorch 1.1.0 working with tensorboard directly (and then came to know Currently it doesn't support anything apart from linear graphs)
<img src="https://github.com/bigmb/Unet-Segmentation-Pytorch-Nest-of-Unets/blob/master/images/tensorb.png" width="280">

**Input Image Visulization for checking**

**a) Original Image**

<img src="https://github.com/bigmb/Unet-Segmentation-Pytorch-Nest-of-Unets/blob/master/images/in1.png" width="480">

**b) CenterCrop Image**

<img src="https://github.com/bigmb/Unet-Segmentation-Pytorch-Nest-of-Unets/blob/master/images/in2.png" width="480">

## 6. Results

**Dice Score for hippocampus segmentation**
ADNI-LONI Dataset

<img src="https://github.com/bigmb/Unet-Segmentation-Pytorch-Nest-of-Unets/blob/master/dice.png" width="380">

## 7. Citation

If you find it usefull for your work. 
```
@article{DBLP:journals/corr/abs-1906-07160,
  author    = {Malav Bateriwala and
               Pierrick Bourgeat},
  title     = {Enforcing temporal consistency in Deep Learning segmentation of brain
               {MR} images},
  journal   = {CoRR},
  volume    = {abs/1906.07160},
  year      = {2019},
  url       = {http://arxiv.org/abs/1906.07160},
  archivePrefix = {arXiv},
  eprint    = {1906.07160},
  timestamp = {Mon, 24 Jun 2019 17:28:45 +0200},
  biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1906-07160},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

## 8. Blog about different Unets
```
In progress
```



# Unet-Segmentation-Pytorch-Nest-of-Unets
Implementation of different kinds of Unet Models for Image Segmentation

1) **UNet** - U-Net: Convolutional Networks for Biomedical Image Segmentation
https://arxiv.org/abs/1505.04597

2) **RCNN-UNet** - Recurrent Residual Convolutional Neural Network based on U-Net (R2U-Net) for Medical Image Segmentation
https://arxiv.org/abs/1802.06955

3) **Attention Unet** - Attention U-Net: Learning Where to Look for the Pancreas
https://arxiv.org/abs/1804.03999

4) **RCNN-Attention Unet** - Attention R2U-Net : Just integration of two recent advanced works (R2U-Net + Attention U-Net)
LeeJun Implementation - https://github.com/LeeJunHyun/Image_Segmentation.git

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
  
  ## 3. Types of Unet
  
  **Unet**
  ![unet1](/images/unet1.png)

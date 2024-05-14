# SSCU-Net
Shallow Super Convolution U-shaped Net

# Model Description
Medical image semantic segmentation plays a vital role in the field of medical image processing by accurately identifying various tissue structures and pathological regions within an image, thereby aiding diagnostic procedures. As the quest for greater accuracy leads to more complex network architectures, it introduces considerable training challenges. To address this, our study presents a new approach with the Shallow Super Convolution U-shaped Net (SSCU-Net), which adopts a minimalist and flattened design. Unlike the conventional four-layer U-shaped networks, the SSCU-Net utilizes a more streamlined two-layer configuration, aligning with our goal of a lightweight design. To compensate for the reduced depth’s potential limitation in semantic feature extraction, we introduce a parallel multi-branch module named the Super Convolution Block (SC-Block) that efficiently captures a wide range of semantic details. Additionally, the Spatial Convolution Path (SC-Path) along with the Feature Enhanced Downsample (FED) and Feature Resolution Upsample (FRU) mechanisms are integrated to ensure effective transmission of crucial semantic information both within and across the network layers. The effectiveness of SSCU-Net was assessed against 18 competing models using seven different metrics across five datasets. Through metric evaluations, visual comparisons, and ablation studies, SSCU-Net has shown a significant performance enhancement, with an average increase of 22.0240% in the Dice coefficient over the other models, thus proving its superiority in combining lightweight architecture with enhanced accuracy.

# Model Architecture

![7295788048a638797ba1a15d90929d06_720](https://github.com/YF-W/SSCU-Net/assets/66008255/365b318b-89f4-4c65-82fe-9e09b64b2d95)


# Env

IDE:	Pycharm 2020.1 Professional ED.

Language:	Python 3.8.15

Framework:	Pytorch 1.13.0

CUDA:	Version 11.7 

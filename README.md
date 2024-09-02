# Neural Collapse Phenomenon Reproduction

This project reproduces the Neural Collapse phenomenon observed during neural network training. The code is based on the ideas and definitions provided in the [paper](https://arxiv.org/pdf/2311.07444). The experiments are conducted on the CIFAR-10 dataset using ResNet18 and VGG11 models.

## Instructions

To compute and visualize the Neural Collapse (NC) metrics during the training process, use the following command:

```bash
python train_cifar.py --epoch 400 --batch_size 64 --gpu 0 --dataset cifar --lr 0.001 --model resnet18
```
## Requirements
* Python 3.10  
* PyTorch 2.0.1  
* Torchvision 0.15.2  

## Dataset
* CIFAR-10
## Models
* ResNet18  
* VGG11
***
This project aims to provide a clear and replicable implementation for understanding the Neural Collapse phenomenon in deep learning.
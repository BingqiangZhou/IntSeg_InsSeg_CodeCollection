# 交互式图像分割、图像实例分割代码合集（持续更新中...）

这里记录一下上一个月(202104)跑通的交互式图像分割、图像实例分割的代码的过程。

## 所做的主要工作

在一些交互式图像分割、图像实例分割工作相关代码（有相应的Github库）的基础上，写出一个推理的类（net.py），然后再Pascal VOC 2012数据集（[官方地址 host.robots.ox.ac.uk](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)、[Pascal VOC Dataset Mirror (pjreddie.com)包含测试集](https://pjreddie.com/projects/pascal-voc-dataset-mirror/))上，测试（test.py）得到分割结果（mask）。

- net.py：预测推理，包含图像预处理（normalize等）、后处理（二值化等）。
- test.py：在Pascal VOC 2012数据集上，做测试得到分割结果（mask）

## 交互式图像分割

### 相关综述

**A survey of recent interactive image segmentation methods**, [PDF - springer.com](https://link.springer.com/content/pdf/10.1007/s41095-020-0177-5.pdf)

### 相关代码

| 网络 | 来源 | 本库中地址 | 打包下载地址 | 相关描述 |
| :-----: | :-----: | :-----: | :-----: | :-----: |


待更新......

## 图像实例分割

### 相关综述

**A Survey on Instance Segmentation: State of the art**, [PDF - arcix.org](https://arxiv.org/pdf/2007.00047)

### 相关代码


| 网络模型 | 来源 | 本库中地址 | 相关描述 |  |
| :---: | :---: | :---: | :---: | :---: |
| MaskRCNN | [torchvision](https://pytorch.org/vision/stable/models.html#mask-r-cnn) | [ImageInstanceSegmentation/MaskRCNN](./ImageInstanceSegmentation/MaskRCNN) | - |[打包下载](https://github.com/BingqiangZhou/IntSeg_InsSeg_CodeCollection/releases/tag/maskrcnn) |
| 待更新......| | | | | 

### 实验环境

windows 10 (20H2)

CUDA 10.2.89

cuDNN 7.6.5

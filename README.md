# 交互式图像分割、图像实例分割代码合集（持续更新中...）

这里记录一下跑通的一些交互式图像分割、图像实例分割的代码。

## 一、主要内容

### 1.1 图像实例分割

在一些有给出相关代码（有相应的Github库）的基础上，封装一个推理预测的类（net.py），然后在Pascal VOC 2012数据集（[官方地址 host.robots.ox.ac.uk](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)、[Pascal VOC Dataset Mirror (pjreddie.com)包含测试集](https://pjreddie.com/projects/pascal-voc-dataset-mirror/))上，测试（test.py）得到分割结果（mask），并计算相关指标（IOU、F1-Score）。

- net.py：预测推理，包含图像预处理（normalize等）、后处理（二值化等）。
- test.py：在Pascal VOC 2012数据集上，做测试得到分割结果（mask），并计算相关指标（IOU、F1-Score）。

### 1.2 交互式图像分割

封装一个推理预测的类（`net.py`），并通过用户交互进行分割测试（`test.py`）。

## 二、[交互式图像分割](./InteractiveImageSegmentation)

### 2.1 相关综述

**A survey of recent interactive image segmentation methods**, [PDF - springer.com](https://link.springer.com/content/pdf/10.1007/s41095-020-0177-5.pdf)

### 2.2 相关代码

#### 2.2.1 相关交互式图像分割方法代码

| 网络 | 来源/源Github库 | 本库中地址 | 相关描述 | |
| :-----: | :-----: | :-----: | :-----: | :-----: |
| DeepGrabCut | [jfzhang95/DeepGrabCut](https://github.com/jfzhang95/DeepGrabCut-PyTorch) | [DeepGrabCut](./InteractiveImageSegmentation/DeepGrabCut) | - | [打包下载](https://github.com/BingqiangZhou/IntSeg_InsSeg_CodeCollection/releases/tag/deepgrabcut) |
| DEXTR | [scaelles/DEXTR](https://github.com/scaelles/DEXTR-PyTorch) | [DEXTR](./InteractiveImageSegmentation/DEXTR) | - | [打包下载](https://github.com/BingqiangZhou/IntSeg_InsSeg_CodeCollection/releases/tag/dextr) |
| G99AIS | [MarcoForte/DeepInteractiveSegmentation](https://github.com/MarcoForte/DeepInteractiveSegmentation) | [G99AIS](./InteractiveImageSegmentation/G99AIS) | - | [打包下载](https://github.com/BingqiangZhou/IntSeg_InsSeg_CodeCollection/releases/tag/g99ais) |
| IIS-LD | [intel-isl/Intseg](https://github.com/intel-isl/Intseg) | [IIS-LD](./InteractiveImageSegmentation/IIS-LD) | Tensorflow 1.x | [打包下载](https://github.com/BingqiangZhou/IntSeg_InsSeg_CodeCollection/releases/tag/iisld) |
| FCTSFN | [cyh4/FCTSFN](https://github.com/cyh4/FCTSFN) | [FCTSFN](./InteractiveImageSegmentation/FCTSFN) | Caffe(通过OpenCV dnn实现推理预测) | [打包下载](https://github.com/BingqiangZhou/IntSeg_InsSeg_CodeCollection/releases/tag/fctsfn) |

待更新......

#### 2.2.2 随机采样

一些根据图像标签随机采点的[随机采样方法](./InteractiveImageSegmentation/RandomSample/random_sample.py)

## 三、[图像实例分割](./ImageInstanceSegmentation)

### 3.1 相关综述

**A Survey on Instance Segmentation: State of the art**, [PDF - arcix.org](https://arxiv.org/pdf/2007.00047)

### 3.2 相关代码

| 网络 | 来源/源Github库 | 修改后的代码 | 相关描述 |  |
| :---: | :---: | :---: | :---: | :---: |
| RSIS | [imatge-upc/rsis](https://github.com/imatge-upc/rsis) | [RSIS](./ImageInstanceSegmentation/RSIS) | - | [打包下载](https://github.com/BingqiangZhou/IntSeg_InsSeg_CodeCollection/releases/tag/rsis) |
| EmbedMask | [yinghdb/EmbedMask](https://github.com/yinghdb/EmbedMask) | [EmbedMask](./ImageInstanceSegmentation/EmbedMask)  | - | [打包下载](https://github.com/BingqiangZhou/IntSeg_InsSeg_CodeCollection/releases/tag/embedmask) |
| RPEIG | <a href="https://github.com/aimerykong/Recurrent-Pixel-Embedding-for-Instance-Grouping">aimerykong/</br>Recurrent-Pixel-Embedding-for-Instance-Grouping</a> | [RPEIG](./ImageInstanceSegmentation/RPEIG) | Matlab代码 | [打包下载](https://github.com/BingqiangZhou/IntSeg_InsSeg_CodeCollection/releases/tag/rpeig) |
| MaskRCNN | [torchvision](https://pytorch.org/vision/stable/models.html#mask-r-cnn) | [MaskRCNN](./ImageInstanceSegmentation/MaskRCNN) | - |[打包下载](https://github.com/BingqiangZhou/IntSeg_InsSeg_CodeCollection/releases/tag/maskrcnn) |
| RefineMask | [zhanggang001/RefineMask](https://github.com/zhanggang001/RefineMask) | [RefineMask](./ImageInstanceSegmentation/RefineMask) | - | [打包下载](https://github.com/BingqiangZhou/IntSeg_InsSeg_CodeCollection/releases/tag/refinemask) |
| CenterMask | [youngwanLEE/CenterMask](https://github.com/youngwanLEE/CenterMask) | [CenterMask](./ImageInstanceSegmentation/CenterMask) | - |[打包下载](https://github.com/BingqiangZhou/IntSeg_InsSeg_CodeCollection/releases/tag/centermask) |
| D2Det | [JialeCao001/D2Det](https://github.com/JialeCao001/D2Det) | [D2Det](./ImageInstanceSegmentation/D2Det) | - |[打包下载](https://github.com/BingqiangZhou/IntSeg_InsSeg_CodeCollection/releases/tag/d2det) |

## 四、实验环境

windows 10 (20H2)

VS Community 2017 (15.9.36)

Matlab 2018b

CUDA 10.2.89 / 11.2.142，[下载地址](https://developer.nvidia.cn/cuda-toolkit-archive)

cuDNN 7.6.5 / 8.0.5，[下载地址](https://developer.nvidia.com/rdp/cudnn-archive)

MinGW-W64 GCC-5.4.0，[下载地址](https://sourceforge.net/projects/mingw-w64/files/mingw-w64/)

----

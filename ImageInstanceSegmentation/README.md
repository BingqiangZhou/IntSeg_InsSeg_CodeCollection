# 图像实例分割

这里记录查看过的一些实例分割代码，包括Github库、论文地址以及所作的一些尝试。

## 代码状态

* ✅  通过pytorch实现推理过程
* ✳️  通过OpenCV DNN模块读取模型以及其他方式（非pytorch）实现推理过程
* ⏹  有代码，没有模型，没有做
* ❌  存在某种困难，战略性放弃
* 🔶  暂时未尝试

## 代码分类

这里主要按以下几种方法分类：

- [基于RNN(循环神经网络)的方法]()
- [基于Embedding的方法](#基于Embedding的方法)
- [基于Detection(检测)的方法](#基于Detection的方法)
- [其他方法](#其他方法)

### 基于RNN的方法

✅ **Recurrent Neural Networks for Semantic Instance Segmentation -arxiv2017**

- [源Github - https://github.com/imatge-upc/rsis](https://github.com/imatge-upc/rsis)
- [Paper PDF - https://arxiv.org/pdf/1712.00617](https://arxiv.org/pdf/1712.00617)
- 支持VOC、Cityscapes 、CVPPP数据集，提供有模型
- Release地址（待发布）
- 实现过程以及记录（待更新）

⏹ **End-to-End Instance Segmentation with Recurrent Attention -CVPR2017**

* [源Github - https://github.com/renmengye/rec-attend-public](https://github.com/renmengye/rec-attend-public)
* [Paper PDF - https://arxiv.org/pdf/1605.09410](https://arxiv.org/pdf/1605.09410)
* 支持Cityscapes 、KITTI 、CVPPP、COCO数据集，未给出模型文件

❌ **Recurrent Instance Segmentation - ECCV2016**

* [源Github - https://github.com/bernard24/RIS](https://github.com/bernard24/RIS)
* [Paper PDF - https://www.robots.ox.ac.uk/~tvg/publications/2016/RIS7.pdf](https://www.robots.ox.ac.uk/~tvg/publications/2016/RIS7.pdf)
* 支持VOC、COCO数据集，有模型，但是Torch(不是Pytorch)的代码(LUA语言)，并且模型文件为`.model`后缀，无法用Pytorch、OpenCV直接加载，这里我放弃了

🔶 RVOS: End-to-End Recurrent Net for Video Object Segmentation -CVPR2019

* [源Github - https://github.com/imatge-upc/rvos](https://github.com/imatge-upc/rvos)
* [Paper PDF - https://arxiv.org/pdf/1903.05612](https://arxiv.org/pdf/1903.05612)
* 基于RNN的方法，扩展到这篇视频多目标分割论文

### 基于Embedding的方法

1. Semantic Instance Segmentation with a Discriminative Loss Function -CVPR2017
   （CVPPP数据集，未给出模型文件）
   https://github.com/Wizaron/instance-segmentation-pytorch
   (instance-segmentation-pytorch-master.zip)
2. Semantic Instance Segmentation via Deep Metric Learning -2017
   （COCO、VOC数据集，未给出模型文件）
   https://github.com/alicranck/instance-seg (instance-seg-master.zip)
3. EmbedMask: Embedding Coupling for One-stage Instance Segmentation -CVPR2019
   （未完成，COCO数据集，有模型，EmbedMask依赖的FCOS编译出错，环境可能需要设置为install.md中的一样）
   https://github.com/yinghdb/EmbedMask (EmbedMask-master.zip)
4. Instance segmentation by jointly optimizing spatial embeddings and clustering bandwidth -CVPR2019
   （Cityscapes数据集，有模型）
   https://github.com/davyneven/SpatialEmbeddings (SpatialEmbeddings-master.zip)
5. Recurrent Pixel Embedding for Instance Grouping -CVPR2018
   （完成，COCO、VOC数据集，有模型，matlab代码得到的embedding特征图，python读取embedding，聚类，匈牙利匹配得到mask）
   https://github.com/aimerykong/Recurrent-Pixel-Embedding-for-Instance-Grouping (matlab)
   (Recurrent-Pixel-Embedding-for-Instance-Grouping-master.zip)

### 基于Detection的方法

1. Mask R-CNN (代码版本众多) -CVPR2017
   （待完成，使用pytorcn中Mask R-CNN模型）
   https://github.com/facebookresearch/maskrcnn-benchmark
   https://github.com/multimodallearning/pytorch-mask-rcnn
   https://github.com/matterport/Mask_RCNN
   https://github.com/bleakie/MaskRCNN
   https://github.com/fizyr/keras-maskrcnn
2. RefineMask: Towards High-Quality Instance Segmentation with Fine-Grained Features -CVPR 2021
   （完成）
   https://github.com/zhanggang001/RefineMask  (RefineMask-main.zip)
3. Weakly-supervised Instance Segmentation via Class-agnostic Learning with Salient Images -CVPR2021(这篇引入了显著图)
   （跳过这一篇，这一篇是通过用粗糙的数据集（包括box以及显著性图），通过学习得到进度比较高的mask，然后用于训练mask-r-cnn）
   https://github.com/vealocia/BoxCaseg (BoxCaseg-main.zip)
4. D2Det: Towards High Quality Object Detection and Instance Segmentation -CVPR2020
   （完成）
   https://github.com/JialeCao001/D2Det-mmdet2.1
   https://github.com/JialeCao001/D2Det (D2Det-master.zip)
5. CenterMask : Real-Time Anchor-Free Instance Segmentation -CVPR2020
   （完成）
   https://github.com/youngwanLEE/CenterMask (CenterMask-master.zip)

### 其他方法

1. Zero-Shot Instance Segmentation  -CVPR2021
   https://github.com/zhengye1995/Zero-shot-Instance-Segmentation (Zero-shot-Instance-Segmentation-main.zip)
2. Weakly Supervised Instance Segmentation using Class Peak Response - CVPR2018
   https://github.com/ZhouYanzhao/PRM/tree/pytorch （PRM-master.zip）
3. Object Counting and Instance Segmentation with Image-level Supervision -CVPR2019
   https://github.com/GuoleiSun/CountSeg (CountSeg-master.zip)

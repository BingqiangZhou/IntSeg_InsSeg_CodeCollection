# 图像实例分割

这里记录查看过的一些实例分割代码，包括Github库、论文地址以及所作的一些尝试。

## 代码状态

* ✅  通过pytorch实现推理过程
* ✳️  通过OpenCV DNN模块读取模型以及其他方式（非pytorch）实现推理过程
* ⏹  有代码，没有模型，没有做
* ❌  存在某种困难或者其他原因，战略性放弃
* 🔶  暂时未尝试

## 代码分类

这里主要按以下几种方法分类：

- [基于RNN(循环神经网络)的方法](#基于RNN的方法)
- [基于Embedding的方法](#基于Embedding的方法)
- [基于Detection(检测)的方法](#基于Detection的方法)
- [其他方法](#其他方法)

### 基于RNN的方法

✅ **Recurrent Neural Networks for Semantic Instance Segmentation -arxiv2017**

- [源Github - https://github.com/imatge-upc/rsis](https://github.com/imatge-upc/rsis)
- [Paper PDF - https://arxiv.org/pdf/1712.00617](https://arxiv.org/pdf/1712.00617)
- 有提供模型文件
- Release地址（待发布）
- 实现过程以及记录（待更新）

⏹ **End-to-End Instance Segmentation with Recurrent Attention -CVPR2017**

* [源Github - https://github.com/renmengye/rec-attend-public](https://github.com/renmengye/rec-attend-public)
* [Paper PDF - https://arxiv.org/pdf/1605.09410](https://arxiv.org/pdf/1605.09410)
* 未给出模型文件

❌ **Recurrent Instance Segmentation - ECCV2016**

* [源Github - https://github.com/bernard24/RIS](https://github.com/bernard24/RIS)
* [Paper PDF - https://www.robots.ox.ac.uk/~tvg/publications/2016/RIS7.pdf](https://www.robots.ox.ac.uk/~tvg/publications/2016/RIS7.pdf)
* 有提供模型文件，但是代码使用的框架是Torch(LUA语言，不是Pytorch)，并且模型文件为`.model`后缀，无法用Pytorch、OpenCV等其他方式加载，这里我放弃了

🔶 **RVOS: End-to-End Recurrent Net for Video Object Segmentation -CVPR2019**

* [源Github - https://github.com/imatge-upc/rvos](https://github.com/imatge-upc/rvos)
* [Paper PDF - https://arxiv.org/pdf/1903.05612](https://arxiv.org/pdf/1903.05612)
* 基于RNN的方法，扩展到这篇视频多目标分割论文

### 基于Embedding的方法

⏹ **Semantic Instance Segmentation with a Discriminative Loss Function -CVPR2017**

* [源Github - https://github.com/Wizaron/instance-segmentation-pytorch](https://github.com/Wizaron/instance-segmentation-pytorch)
* [Paper PDF - https://arxiv.org/pdf/1708.02551](https://arxiv.org/pdf/1708.02551.pdf)
* 未给出模型文件

⏹ **Semantic Instance Segmentation via Deep Metric Learning -2017**

* [源Github - https://github.com/alicranck/instance-seg](https://github.com/alicranck/instance-seg)
* [Paper PDF - https://arxiv.org/pdf/1703.10277](https://arxiv.org/pdf/1703.10277)
* 未给出模型文件

✅ **EmbedMask: Embedding Coupling for One-stage Instance Segmentation -CVPR2019**

* [源Github - https://github.com/yinghdb/EmbedMask](https://github.com/yinghdb/EmbedMask)
* [Paper PDF - https://arxiv.org/pdf/1912.01954](https://arxiv.org/pdf/1912.01954)
* 有提供模型文件
* Release地址（待发布）
* 实现过程以及记录（待更新）

🔶 **Instance segmentation by jointly optimizing spatial embeddings and clustering bandwidth -CVPR2019**

* [源Github - https://github.com/davyneven/SpatialEmbeddings](https://github.com/davyneven/SpatialEmbeddings)
* [Paper PDF - https://arxiv.org/pdf/1906.11109](https://arxiv.org/pdf/1906.11109)
* 有提供模型文件
* 只提供了在Cityscapes数据集上训练得到的模型，Cityscapes数据集与Pascal VOC 2012数据集有较大差异，所以没有尝试

✅ **Recurrent Pixel Embedding for Instance Grouping -CVPR2018**

* [源Github - https://github.com/aimerykong/Recurrent-Pixel-Embedding-for-Instance-Grouping](https://github.com/aimerykong/Recurrent-Pixel-Embedding-for-Instance-Grouping)
* [Paper PDF - https://arxiv.org/pdf/1712.08273](https://arxiv.org/pdf/1712.08273)
* 有提供模型文件，matlab代码得到的embedding特征图，python读取embedding，聚类，匈牙利匹配得到mask
* Release地址（待发布）
* 实现过程以及记录（待更新）

### 基于Detection的方法

✅ **Mask R-CNN -CVPR2017**

* [Paper PDF - https://arxiv.org/pdf/1703.06870](https://arxiv.org/pdf/1703.06870)
* 相关代码有许多，如下：
  - [https://github.com/facebookresearch/maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark)
  - [https://github.com/multimodallearning/pytorch-mask-rcnn](https://github.com/multimodallearning/pytorch-mask-rcnn)
  - [https://github.com/matterport/Mask_RCNN](https://github.com/matterport/Mask_RCNN)
  - [https://github.com/bleakie/MaskRCNN](https://github.com/bleakie/MaskRCNN)
  - [https://github.com/fizyr/keras-maskrcnn](https://github.com/fizyr/keras-maskrcnn)
* 这里直接通过使用torchvision包中带有的Mask R-CNN([参考文档](https://pytorch.org/vision/stable/models.html#mask-r-cnn))来实现推理预测过程
* [代码打包下载](https://github.com/BingqiangZhou/IntSeg_InsSeg_CodeCollection/releases/tag/maskrcnn)
* [跑通过程记录](./MaskRCNN)

✅ **RefineMask: Towards High-Quality Instance Segmentation with Fine-Grained Features -CVPR 2021**

* [源Github - https://github.com/zhanggang001/RefineMask](https://github.com/zhanggang001/RefineMask)
* [Paper PDF - https://arxiv.org/pdf/2104.08569](https://arxiv.org/pdf/2104.08569.pdf)
* 有提供模型文件
* Release地址（待发布）
* 实现过程以及记录

❌ **Weakly-supervised Instance Segmentation via Class-agnostic Learning with Salient Images -CVPR2021**

* [源Github - https://github.com/vealocia/BoxCaseg](https://github.com/vealocia/BoxCaseg)
* [Paper PDF - https://arxiv.org/pdf/2104.01526](https://arxiv.org/pdf/2104.01526.pdf)
* 跳过这一篇，简单的看了Github中的介绍，这一篇是通过用粗糙的数据集（包括box以及显著性图），通过学习得到精度比较高的mask，然后用于训练mask-r-cnn

✅ **D2Det: Towards High Quality Object Detection and Instance Segmentation -CVPR2020**

* [源Github - https://github.com/JialeCao001/D2Det](https://github.com/JialeCao001/D2Det)
* [另一个版本 - https://github.com/JialeCao001/D2Det-mmdet2.1 - 支持更高的mmdet版本](https://github.com/JialeCao001/D2Det-mmdet2.1)
* [Paper PDF - https://ieeexplore.ieee.org/document/9157372](https://ieeexplore.ieee.org/document/9157372)
* 有提供模型文件
* Release地址（待发布）
* 实现过程以及记录（待更新）

✅ **CenterMask : Real-Time Anchor-Free Instance Segmentation -CVPR2020**

* [源Github - https://github.com/youngwanLEE/CenterMask](https://github.com/youngwanLEE/CenterMask)
* [Paper PDF - https://arxiv.org/pdf/1911.06667](https://arxiv.org/pdf/1911.06667)
* 有提供模型文件
* Release地址（待发布）
* 实现过程以及记录（待更新）

### 其他方法

🔶 **Zero-Shot Instance Segmentation  -CVPR2021**

* [源Github - https://github.com/zhengye1995/Zero-shot-Instance-Segmentation](https://github.com/zhengye1995/Zero-shot-Instance-Segmentation)
* [Paper PDF - https://arxiv.org/pdf/2104.06601](https://arxiv.org/pdf/2104.06601.pdf)

🔶 **Weakly Supervised Instance Segmentation using Class Peak Response - CVPR2018**

* [源Github - https://github.com/ZhouYanzhao/PRM/tree/pytorch](https://github.com/ZhouYanzhao/PRM/tree/pytorch)
* [Paper PDF - https://arxiv.org/pdf/1804.00880](https://arxiv.org/pdf/1804.00880)

🔶 **Object Counting and Instance Segmentation with Image-level Supervision -CVPR2019**

* [源Github - https://github.com/GuoleiSun/CountSeg](https://github.com/GuoleiSun/CountSeg)
* [Paper PDF - https://arxiv.org/pdf/1903.02494v2](https://arxiv.org/pdf/1903.02494v2)

---

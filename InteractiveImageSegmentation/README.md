# 交互式图像分割

这里记录查看过的一些交互式图像代码，包括Github库、论文地址以及所作的一些尝试。

## 代码状态

* ✅  通过pytorch实现推理过程
* ✳️  通过OpenCV DNN模块读取模型以及其他方式（非pytorch）实现推理过程
* ⏹  有代码，没有模型，没有做
* ❌  存在某种困难，战略性放弃
* 🔶  暂时未尝试

## 代码分类

这里主要按年份分类：

* [2018](#2018)
* [2019](#2019)
* [2020](#2020)
* [2021](#2021)

### 2017

Deep extreme cut: From extreme points to object segmentation
https://github.com/scaelles/DEXTR-PyTorch/
http://www.vision.ee.ethz.ch/～cvlsegmentation/dextr/

### 2018

(1 把加载数据集数据部分写进了计算图，战略性放弃)
Iteratively trained interactive segmentation
https://github.com/sabarim/itis

(2 完成)
Interactive image segmentation with latent diversity
https://github.com/intel-isl/Intseg

(3 未给出模型文件)
SeedNet: Automatic seed generation with deep reinforcement learning for robust interactive segmentation.
https://github.com/kelawaad/SeedNet

(4 完成，caffe python源码，有模型，使用opencv dnn模块加载caffe模型进行推理，重新编译opencv，使用CUDA加速)
A fully convolutional two-stream fusion network for interactive image segmentation
https://github.com/cyh4/FCTSFN

### 2019

(5 完成，caffe python源码，有模型，使用opencv dnn模块加载caffe模型进行推理，重新编译opencv，使用CUDA加速)
Interactive image segmentation via backpropagating refinement scheme
https://github.com/wdjang/BRS-Interactive_segmentation

### 2020

(6 未完成)
F-BRS: Rethinking backpropagating refinement for interactive segmentation
https://github.com/saic-vul/fbrs_interactive_segmentation
https://github.com/jpconnel/fbrs-segmentation

(7 主要处理遥感图像)
DISIR: Deep image segmentation with interactive refinement
https://github.com/delair-ai/DISIR

(8 完成)
Getting to 99% accuracy in interactive segmentation
https://github.com/MarcoForte/DeepInteractiveSegmentation

(9 未完成)
Interactive Object Segmentation with Inside-Outside Guidance
https://github.com/shiyinzhang/Inside-Outside-Guidance

### 2021

(10 未完成)

Reviving Iterative Training with Mask Guidance for Interactive Segmentation
https://github.com/saic-vul/ritm_interactive_segmentation

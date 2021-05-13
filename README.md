# 交互式图像分割、图像实例分割代码合集（持续更新中...）

这里记录一下上一个月(202104)跑通的交互式图像分割、图像实例分割的代码的过程。

## 主要做的工作

在一些交互式图像分割、图像实例分割工作相关代码（有相应的Github库）的基础上，写出一个推理的类（net.py），然后再Pascal VOC 2012数据集（[官方地址 host.robots.ox.ac.uk](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)、[Pascal VOC Dataset Mirror (pjreddie.com 包含测试集)](https://pjreddie.com/projects/pascal-voc-dataset-mirror/))上，测试（test.py）得到分割结果（mask）。

- net.py：预测推理，包含图像预处理（normalize等）、后处理（二值化等）。
- test.py：在Pascal VOC 2012数据集上，做测试得到分割结果（mask）

## 交互式图像分割

相关综述 ：xxxx

xxx库，原地址，本库中地址，release

## 图像实例分割

相关综述 ：xxxx

xxx库，原地址，本库中地址，release

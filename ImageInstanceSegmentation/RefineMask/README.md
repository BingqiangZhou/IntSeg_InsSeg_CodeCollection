# RefineMask跑通过程记录

在作者提供的代码([Github地址](https://github.com/zhanggang001/RefineMask))的基础上，进行修改，实现推理预测。

## 创建环境以及安装相关Python包

**注意：首先得安装好CUDA、cuDNN以及GCC，[下载地址](../../README.md#实验环境)**

由于mmcv-full一直编译出错，这里改用了CUDA 11.1、cuDNN 8.0.5，参考：[windows 10 install mmcv-full error](https://github.com/open-mmlab/mmcv/issues/789#issuecomment-768289939)

````bash

# 下载源代码
git clone https://github.com/zhanggang001/RefineMask.git
cd RefineMask

# 创建虚拟环境
conda create -n refinemask python=3.6
conda activate refinemask

# 安装pytorch、torchvision
conda install pytorch torchvision cudatoolkit=11.1 -c pytorch -c nvidia

# 下载依赖相关包
pip install cython yacs matplotlib pandas opencv-python tqdm openpyxl scipy terminaltables pycocotools lvis

# 安装mmcv，对应pytorch的版本见https://github.com/open-mmlab/mmcv#installation，这里会需要编译安装，会花费一二十分钟的时间
pip install mmcv-full

````

## 修改及编写代码
### [代码打包下载](https://github.com/BingqiangZhou/IntSeg_InsSeg_CodeCollection/releases/tag/refinemask)

### 下载模型文件

**需要科学上网** [源Github库](https://github.com/zhanggang001/RefineMask)下提供了许多模型的[下载地址](https://github.com/zhanggang001/RefineMask#main-results)，我将`r50-coco-2x.pth`放在了本仓库的[Releaes](https://github.com/BingqiangZhou/IntSeg_InsSeg_CodeCollection/releases/tag/refinemask)中，提供下载。

这里以`r50-coco-2x.pth`为例，下载后放到`RefineMask\models`文件夹中。

### 修改源码

1. 源码中限制了mmcv的版本为1.0.5，我们这里将其去掉，注释掉[`\mmdet\__init__.py`](https://github.com/BingqiangZhou/IntSeg_InsSeg_CodeCollection/blob/master/ImageInstanceSegmentation/RefineMask/mmdet/__init__.py)中的内容，只保留`import mmcv`这一句。

2. 注释了`mmdet\models\builder.py`中[`build_detector`](https://github.com/BingqiangZhou/IntSeg_InsSeg_CodeCollection/blob/0339cef73ed674245423aad67927406a03c484fc/ImageInstanceSegmentation/RefineMask/mmdet/models/builder.py#L66)方法中Loger相关的代码，不再输出网络结构。

3. 为了记录推理的时间，这里修改了`mmdet\apis\inference.py`中的[inference_detector](https://github.com/BingqiangZhou/IntSeg_InsSeg_CodeCollection/blob/0339cef73ed674245423aad67927406a03c484fc/ImageInstanceSegmentation/RefineMask/mmdet/apis/inference.py#L77)方法，在返回预测结果的同时，返回推理时间。

### net.py

**1. 输入图像预处理**

这里直接输入图片的路径即可，直接调用源代码`mmdet\apis\inference.py`中的[inference_detector](https://github.com/BingqiangZhou/IntSeg_InsSeg_CodeCollection/blob/0339cef73ed674245423aad67927406a03c484fc/ImageInstanceSegmentation/RefineMask/mmdet/apis/inference.py#L77)方法得到结果。

**2. 后处理**

网络的输出结果包括有masks和bboxes
- masks：80个分类（对应coco中的80个对象类别），数据结构为`list[list]`,，每个分类下对应一个list，mask对应的格式是Run Length Encoding (RLE)。
- bboxes：同样是80个类别，对应每一个mask，bbox包括五个值，四个值是bbox坐标，第五个值是分数

这里对上面的输出结果做了如下处理：
1. 给定一个阈值（默认0.5），根据bbox的分数，舍弃小于等于阈值的结果。
2. 将保留下来的结果，形式变为：
    - masks：list[二值图像格式]，数据结果为二值图像的列表
    - bboxes: list[bbox坐标]，这里不包括分数
    - class_labels: list[mask的类别标号]

### test.py

将整个Pascal VOC 2012验证集中的图片跑一遍预测，通过真实的mask与预测的mask做最大匹配，得到一一对应的mask，并求出相关的指标，最后保存mask成图像，相关指标数据保存到Excel当中。

---
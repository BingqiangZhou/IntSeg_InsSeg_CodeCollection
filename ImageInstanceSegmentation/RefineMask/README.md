# D2Det跑通过程记录

在作者提供的代码([Github地址](https://github.com/JialeCao001/D2Det))的基础上，进行修改，实现推理预测。

源代码基于[mmdetection](https://github.com/open-mmlab/mmdetection)实现，应该与[FCOS](https://github.com/tianzhi0549/FCOS)和[maskrcn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark)有某种联系，有一些相似。

## 创建环境以及安装相关Python包

**注意：首先得安装好CUDA、cuDNN以及GCC，[下载地址](../../README.md#实验环境)**

````bash

# 下载源代码
git clone https://github.com/JialeCao001/D2Det.git
cd D2Det

# 创建虚拟环境
conda create -n d2det python=3.7
conda activate d2det

# 安装pytorch(1.4.0)、torchvision(0.5.0)包，在windows下，得安装这个版本才行，其他版本会报错，见连接[RuntimeError: Error compiling objects for extension](https://github.com/facebookresearch/maskrcnn-benchmark/issues/1236#issuecomment-645739809)
# 这里的cudatoolkit版本是10.1，而CUDA的版本是10.2，版本不一样，但是在实践中发现没有太大问题，并且如果直接修改cudatoolkit=10.2，会找不到包
conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch

# 安装mmcv，mmdetection对其有依赖，这里会需要编译安装，会花费十分钟左右的时间
# 注意不要安装过高的版本，安装过高的版本，可能会报错[“No module named 'mmcv.cnn.weight_init’”](https://github.com/open-mmlab/mmdetection/issues/3402#issuecomment-680420003)
pip install mmcv==0.4.3

# 下载安装mmdetection
git clone https://github.com/open-mmlab/mmdetection.git

# 下载安装mmdetection依赖的pycocotools
cd mmdetection
pip install "git+https://github.com/open-mmlab/cocoapi.git#subdirectory=pycocotools"

# 回到源代码的D2Det路径下，编译安装，这里会需要十分钟左右
cd ../
python setup.py build develop

# 可选，test.py相关依赖
pip install pandas opencv-python scipy openpyxl tqdm

````

## 修改及编写代码
### [代码打包下载](https://github.com/BingqiangZhou/IntSeg_InsSeg_CodeCollection/releases/tag/d2det)

### 下载模型文件

**需要科学上网** [源Github库](https://github.com/JialeCao001/D2Det)下提供了许多模型的[下载地址](https://github.com/JialeCao001/D2Det#results)，我将`D2Det-instance-res101.pth`放在了本仓库的[Releaes](https://github.com/BingqiangZhou/IntSeg_InsSeg_CodeCollection/releases/tag/d2det)中，提供下载。

这里以`D2Det-instance-res101.pth`为例，下载后放到`D2Det\models`文件夹中。

### 修改源码

1. 为了记录推理的时间，这里修改了`mmdet\apis\inference.py`中的[inference_detector](https://github.com/BingqiangZhou/IntSeg_InsSeg_CodeCollection/blob/8049d66e67a24f7dbd0d0b0fb23ae8416886dc88/ImageInstanceSegmentation/D2Det/mmdet/apis/inference.py#L63)方法，在返回预测结果的同时，返回推理时间。

### net.py

**1. 输入图像预处理**

这里直接输入图片的路径即可，直接调用源代码`mmdet\apis\inference.py`中的[inference_detector](https://github.com/BingqiangZhou/IntSeg_InsSeg_CodeCollection/blob/8049d66e67a24f7dbd0d0b0fb23ae8416886dc88/ImageInstanceSegmentation/D2Det/mmdet/apis/inference.py#L63)方法得到结果。

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


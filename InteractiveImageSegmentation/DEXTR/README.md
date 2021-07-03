# DEXTR

在作者提供的代码([Github地址](https://github.com/scaelles/DEXTR-PyTorch/))的基础上，实现推理预测。

## 创建环境以及安装相关Python包

**注意：首先得安装好CUDA、cuDNN以及GCC，[下载地址](../../README.md#实验环境)**

````bash
conda install pytorch torchvision -c pytorch
conda install matplotlib opencv pillow scikit-learn scikit-image
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


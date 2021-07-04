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

**可能需要科学上网** [源Github库](https://github.com/scaelles/DEXTR-PyTorch/)下提供了的模型的[下载地址](https://github.com/scaelles/DEXTR-PyTorch/#pre-trained-models)，我将它们放在了本仓库的[Releaes](https://github.com/BingqiangZhou/IntSeg_InsSeg_CodeCollection/releases/tag/dextr)中，提供下载。

以`dextr_pascal-sbd.pth`为例，下载后放到`DEXTR/models`文件夹中。

### [net.py](https://github.com/BingqiangZhou/IntSeg_InsSeg_CodeCollection/blob/master/InteractiveImageSegmentation/DEXTR/net.py)

`net.py`主要提取[源Github库](https://github.com/scaelles/DEXTR-PyTorch/)中[demo.py](https://github.com/scaelles/DEXTR-PyTorch/blob/master/demo.py)文件中的内容，并将其封装起来。

### [test.py](https://github.com/BingqiangZhou/IntSeg_InsSeg_CodeCollection/blob/master/InteractiveImageSegmentation/DEXTR/test.py)

交互分割单张图像，通过OpenCV显示源图像，进行交互，显示分割结果，具体操作有：

1. 运行程序，选择一张图像
2. 左键进行交互（打点），右键撤销上次交互，当完成四个极值点的交互时，可以按下p键("P" or "p")进行分割，交互结果在显示的同时将保存在原图像的同目录下。
3. 按下o键("O" or "o"))，选择一张新的照片进行分割。

---


# DEXTR

在作者提供的代码([Github地址](https://github.com/MarcoForte/DeepInteractiveSegmentation))的基础上，实现推理预测。

## 创建环境以及安装相关Python包

**注意：首先得安装好CUDA、cuDNN以及GCC，[下载地址](../../README.md#实验环境)**

````bash
conda install pytorch torchvision -c pytorch
conda install matplotlib opencv pillow scikit-learn scikit-image guided_filter_pytorch
````

## 修改及编写代码
### [代码打包下载](https://github.com/BingqiangZhou/IntSeg_InsSeg_CodeCollection/releases/tag/g99ais)

### 下载模型文件

**可能需要科学上网** [源Github库](https://github.com/MarcoForte/DeepInteractiveSegmentation)下提供了的模型的[下载地址](https://drive.google.com/file/d/1nJMTXSlprm5FQaQA5gfyU8CbSEX8ghzJ/view?usp=sharing)，我将它们放在了本仓库的[Releaes](https://github.com/BingqiangZhou/IntSeg_InsSeg_CodeCollection/releases/tag/g99ais)中，提供下载。

以`dextr_pascal-sbd.pth`为例，下载后放到`DEXTR/models`文件夹中。

### [net.py](https://github.com/BingqiangZhou/IntSeg_InsSeg_CodeCollection/blob/master/InteractiveImageSegmentation/DeepInteractiveSegmentation/net.py)

`net.py`主要提取[源Github库](https://github.com/MarcoForte/DeepInteractiveSegmentation)中[demo.py](https://github.com/MarcoForte/DeepInteractiveSegmentation/blob/master/demo.py)文件中的内容，并将其封装起来。

### [test.py](https://github.com/BingqiangZhou/IntSeg_InsSeg_CodeCollection/blob/master/InteractiveImageSegmentation/DeepInteractiveSegmentation/test.py)

交互分割单张图像，通过OpenCV显示源图像，进行交互，显示分割结果，具体操作有：

1. 运行程序，选择一张图像
2. 左键进行前景交互（打点），右键进行背景交互，按住'ctrl'键并点击左键可以去掉上一次交互点，当完成交互时，可以按下p键("P" or "p")进行分割，交互结果在显示的同时将保存在原图像的同目录下。
3. 按下o键("O" or "o"))，选择一张新的照片进行分割。

### [test_iter.py](https://github.com/BingqiangZhou/IntSeg_InsSeg_CodeCollection/blob/master/InteractiveImageSegmentation/DeepInteractiveSegmentation/test_iter.py)

迭代分割：打一个点分割一次，上一次的结果将会输入到网络中。

1. 左键进行前景交互（打点），右键进行背景交互，按住'ctrl'键并点击左键可以去掉上一次交互点。
2. 按下o键("O" or "o"))，选择一张新的照片进行分割。
3. 按下c键("c" or "C"))，对当前图片重新开始进行交互、分割。

---


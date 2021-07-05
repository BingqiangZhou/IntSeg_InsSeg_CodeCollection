# BRS

在作者提供的代码([Github地址](https://github.com/wdjang/BRS-Interactive_segmentation))的基础上，实现推理预测。

## 创建环境以及安装相关Python包

**注意：首先得安装好CUDA、cuDNN以及GCC，[下载地址](../../README.md#实验环境)**

OpenCV + Numpy + Pillow

注：这里是通过OpenCV加载Caffe模型实现推理预测，如果需要使用GPU加速的话，需要自己编译OpenCV，相关的文档见[Docs](../../Docs)。

## 修改及编写代码
### [代码打包下载](https://github.com/BingqiangZhou/IntSeg_InsSeg_CodeCollection/releases/tag/brs)

### 下载模型文件

**可能需要科学上网** [源Github库](https://github.com/wdjang/BRS-Interactive_segmentation)下提供了的模型的[下载地址](https://www.dropbox.com/s/o5i2autfzfos1tk/BRS_DenseNet.caffemodel?dl=0)，我将它们放在了本仓库的[Releaes](https://github.com/BingqiangZhou/IntSeg_InsSeg_CodeCollection/releases/tag/brs)中，提供下载。

### [net.py](./net.py)

`net.py`将推理预测的过程封装起来。

### [test_iter.py](./test_iter.py)

迭代分割：打一个点分割一次。

1. 左键进行前景交互（打点），右键进行背景交互，按住'ctrl'键并点击左键可以去掉上一次交互点。
2. 按下o键("o" or "O"))，选择一张新的照片进行分割。
3. 按下c键("c" or "C"))，对当前图片重新开始进行交互、分割。
4. 按下c键("s" or "S"))，保存交互以及分割结果。

### [test_voc.py](./test_voc.py)

在PASCAL VOC 2012数据集上做测试，并记录指标，交互由[随机采样方法](../RandomSample/random_sample.py)采得。

---


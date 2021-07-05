# BRS

在作者提供的代码([Github地址](https://github.com/saic-vul/fbrs_interactive_segmentation))的基础上，实现推理预测。

## 创建环境以及安装相关Python包

**注意：首先得安装好CUDA、cuDNN，[下载地址](../../README.md#实验环境)**

请看[requirements.txt](./requirements.txt)，并不是所有的库，都需要，可以直接运行，缺什么库补什么库。

## 修改及编写代码
### [代码打包下载](https://github.com/BingqiangZhou/IntSeg_InsSeg_CodeCollection/releases/tag/fbrs)

### 下载模型文件

**可能需要科学上网** [源Github库](https://github.com/saic-vul/fbrs_interactive_segmentation)下`release`提供了的模型的[下载地址](https://github.com/saic-vul/fbrs_interactive_segmentation#pretrained-models)。

### [net.py](./net.py)

`net.py`将推理预测的过程封装起来。

### [test_iter.py](./test_iter.py)

迭代分割：打一个点分割一次。

1. 左键进行前景交互（打点），右键进行背景交互，按住'ctrl'键并点击左键可以去掉上一次交互点。
2. 按下o键("o" or "O"))，选择一张新的照片进行分割。
3. 按下c键("c" or "C"))，对当前图片重新开始进行交互、分割。
4. 按下c键("s" or "S"))，保存交互以及分割结果。

---


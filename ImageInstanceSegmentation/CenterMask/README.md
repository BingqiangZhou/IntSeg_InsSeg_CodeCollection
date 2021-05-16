# CenterMask跑通过程记录

在作者提供的代码([Github地址](https://github.com/youngwanLEE/CenterMask))的基础上，进行修改，实现推理预测。

源代码基于[FCOS](https://github.com/tianzhi0549/FCOS)和[maskrcn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark)实现。

## 创建环境以及安装相关Python包

**注意：首先得安装好CUDA、cuDNN以及GCC，[下载地址](../../README.md#实验环境)**

````bash

# 下载源代码
git clone https://github.com/youngwanLEE/CenterMask.git
cd CenterMask

# 创建虚拟环境
conda create -n centermask python=3.7
conda activate centermask

# this installs the right pip and dependencies for the fresh python
# 这一步不知道是不是一定需要，但是源代码中的Install.md中有这一步，这里就直接复制过来执行了
conda install ipython

# 安装pytorch(1.4.0)、torchvision(0.5.0)包，主要要windows下，得安装这个版本才行，其他版本会报错，见连接[RuntimeError: Error compiling objects for extension](https://github.com/facebookresearch/maskrcnn-benchmark/issues/1236#issuecomment-645739809)
# 这里的cudatoolkit版本是10.1，而CUDA的版本是10.2，版本不一样，但是在实践中发现没有太大问题，并且如果直接修改cudatoolkit=10.2，会找不到包
conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch

# 下载依赖相关包
pip install ninja yacs cython matplotlib tqdm 

# 可选，test.py相关依赖
pip install pandas opencv-python scipy openpyxl

# 下载安装pycocotools
cd CenterMask
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install
# 这里需要注意，在windows下安装会报错（cl: 命令行 error D8021 :无效的数值参数“/Wno-cpp”）
# 注释掉cocoapi\PythonAPI\setup.py文件中的 extra_compile_args=['-Wno-cpp', '-Wno-unused-function', '-std=c99'] 这一句，
# 并且注意删掉cocoapi\PythonAPI\build文件夹，再重新编译

# 回到源代码的CenterMask路径下，编译安装，这里会需要几分钟
cd ../../
python setup.py build develop

````

## 修改及编写代码
### [代码打包下载](https://github.com/BingqiangZhou/IntSeg_InsSeg_CodeCollection/releases/tag/centermask)

### 下载模型文件

**需要科学上网** [源Github库](https://github.com/youngwanLEE/CenterMask)下提供了许多模型的[下载地址](https://github.com/youngwanLEE/CenterMask#models)，我将`centermask-R-50-FPN-ms-2x.pth`放在了本仓库的[Releaes](https://github.com/BingqiangZhou/IntSeg_InsSeg_CodeCollection/releases/tag/centermask)中，提供下载。

这里以[centermask-R-50-FPN-ms-2x.pth](https://www.dropbox.com/s/bhpf6jud8ovvxmh/centermask-R-50-FPN-ms-2x.pth?dl=1)为例，下载后放到`CenterMask\models`文件夹中

### 修改CenterMask源码

1. 为了记录推理的时间，这里修改了`demo\predictor.py`中的[compute_prediction](https://github.com/BingqiangZhou/IntSeg_InsSeg_CodeCollection/blob/7f614785c5afc42d4570e0d5e2fbbcbc37219e28/ImageInstanceSegmentation/CenterMask/demo/predictor.py#L240)方法，在返回预测结果的同时，返回推理时间。

### net.py

**1. 输入图像预处理**

这里直接输入图片的路径即可，直接调用源代码`demo\predictor.py`中的[compute_prediction](https://github.com/BingqiangZhou/IntSeg_InsSeg_CodeCollection/blob/7f614785c5afc42d4570e0d5e2fbbcbc37219e28/ImageInstanceSegmentation/CenterMask/demo/predictor.py#L240)方法得到结果。

**2. 后处理**

网络输出包括有mask([50, 1, h, w]，值为0或者1)，mask分数scores([50])，边框boxes([50, 4])，类别标签labels([50])，这里只保留mask分数大于阈值的mask、mask分数scores，边框boxes，类别标签labels，随后将其转为ndarray保存在列表中(其中mask的大小为[h, w]没有多的维度)。

### test.py

将整个Pascal VOC 2012验证集中的图片跑一遍预测，通过真实的mask与预测的mask做最大匹配，得到一一对应的mask，并求出相关的指标，最后保存mask成图像，相关指标数据保存到Excel当中。

---


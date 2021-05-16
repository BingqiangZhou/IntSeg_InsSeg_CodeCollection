# EmbedMask跑通过程记录

在作者提供的代码([Github地址](https://github.com/yinghdb/EmbedMask))的基础上，进行修改，实现推理预测。

源代码基于[FCOS](https://github.com/tianzhi0549/FCOS)和[maskrcn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark)实现。

## 创建环境以及安装相关Python包

**注意：首先得安装好CUDA、cuDNN以及GCC，[下载地址](../../README.md#实验环境)**

````bash

# 下载源代码
git clone https://github.com/yinghdb/EmbedMask.git
cd EmbedMask

# 创建虚拟环境
conda create -n embedmask python=3.6
conda activate embedmask

# this installs the right pip and dependencies for the fresh python
# 这一步不知道是不是一定需要，但是源代码中的Install.md中有这一步，这里就直接复制过来执行了
conda install ipython

# 安装pytorch(1.4.0)、torchvision(0.5.0)包，主要要windows下，得安装这个版本才行，其他版本会报错，见连接[RuntimeError: Error compiling objects for extension](https://github.com/facebookresearch/maskrcnn-benchmark/issues/1236#issuecomment-645739809)
# 这里的cudatoolkit版本是10.1，而CUDA的版本是10.2，版本不一样，但是在实践中发现没有太大问题，并且如果直接修改cudatoolkit=10.2，会找不到包
conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch

# 下载依赖相关包
pip install ninja yacs cython matplotlib tqdm pandas opencv-python scipy openpyxl scikit-image 

# 后面的步骤不需要，即可运行

# # 下载安装pycocotools
# git clone https://github.com/cocodataset/cocoapi.git
# cd cocoapi/PythonAPI
# python setup.py build_ext install
# # 这里需要注意，在windows下安装会报错（cl: 命令行 error D8021 :无效的数值参数“/Wno-cpp”）
# # 注释掉cocoapi\PythonAPI\setup.py文件中的 extra_compile_args=['-Wno-cpp', '-Wno-unused-function', '-std=c99'] 这一句，
# # 并且注意删掉cocoapi\PythonAPI\build文件夹，再重新编译

# # 回到源代码的CenterMask路径下，编译安装，这里会需要几分钟
# cd ../../
# python setup.py build develop

````

## 修改及编写代码
### [代码打包下载](https://github.com/BingqiangZhou/IntSeg_InsSeg_CodeCollection/releases/tag/embedmask)

### 下载模型文件

**需要科学上网** [源Github库](https://github.com/yinghdb/EmbedMask)下提供了预训练模型的[下载地址](https://github.com/yinghdb/EmbedMask#pretrained-models)，我将作者给出的预训练模型`embed_mask_R50_1x.pth`和`embed_mask_R101_ms_3x.pth`放在了本仓库的[Releaes](https://github.com/BingqiangZhou/IntSeg_InsSeg_CodeCollection/releases/tag/embedmask)中，提供下载。

### 修改源码

1. 不再将每个分类的的置信分数的阈值作为参数，直接将去放在[`COCODemo`](./demo/predictor.py)类中`confidence_thresholds_for_classes`，见[`demo/predictor.py`](https://github.com/BingqiangZhou/IntSeg_InsSeg_CodeCollection/blob/943a042f9d028ef8242829baa7c8db6b2dd9bf28/ImageInstanceSegmentation/EmbedMask/demo/predictor.py#L129)。

2. 预处理去掉Resize操作，见[`demo/predictor.py`](https://github.com/BingqiangZhou/IntSeg_InsSeg_CodeCollection/blob/943a042f9d028ef8242829baa7c8db6b2dd9bf28/ImageInstanceSegmentation/EmbedMask/demo/predictor.py#L181)。

3. 加入记录推理的时间，修改`demo/predictor.py`中的[compute_prediction](https://github.com/BingqiangZhou/IntSeg_InsSeg_CodeCollection/blob/943a042f9d028ef8242829baa7c8db6b2dd9bf28/ImageInstanceSegmentation/EmbedMask/demo/predictor.py#L234)方法，在返回预测结果的同时，返回推理时间。

### net.py

**1. 输入图像预处理**

这里读取到图片，然后输入图片即可，调用源代码`demo/predictor.py`中的[compute_prediction](https://github.com/BingqiangZhou/IntSeg_InsSeg_CodeCollection/blob/943a042f9d028ef8242829baa7c8db6b2dd9bf28/ImageInstanceSegmentation/EmbedMask/demo/predictor.py#L217)方法得到结果。

**2. 后处理**

只保留预测mask的分数，大于阈值（默认0.2）的相关数据，包括mask、bbox、类别标签、mask的分数。

### test.py

将整个Pascal VOC 2012验证集中的图片跑一遍预测，通过真实的mask与预测的mask做最大匹配，得到一一对应的mask，并求出相关的指标，最后保存mask成图像，相关指标数据保存到Excel当中。

---
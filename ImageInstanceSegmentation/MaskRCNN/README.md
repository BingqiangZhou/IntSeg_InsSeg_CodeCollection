# Mask R-CNN跑通过程记录

使用torchvision中自带的mask r-cnn模型实现，[参考文档 - torchvision.models.detection.maskrcnn_resnet50_fpn](https://pytorch.org/vision/stable/models.html#mask-r-cnn)

## 创建环境以及安装相关Python包

````bash
# 创建虚拟环境
conda create -n maskrcnn python=3.7
conda activate maskrcnn

# 安装pytorch(当前版本：1.8.1)、torchvision(当前版本：0.9.1)
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch

# 为test.py安装相关包
pip install numpy pillow matplotlib pandas tqdm opencv-python scipy openpyxl

````

## 修改及编写代码
### [代码打包下载](https://github.com/BingqiangZhou/IntSeg_InsSeg_CodeCollection/releases/tag/maskrcnn)

### [net.py](./net.py)

**1. 输入图像预处理**

这里只需要将图像转为(1, C, H, W)大小即可。文档中输入为(C, H, W)的list，但(N, C, H, W)也可以。

**2. 后处理**

网络输出为`List[Dict[Tensor]]`，包括有mask，分数scores，边框boxes，类别标签labels，这里将其mask置为二值图，并将所有输出都转为ndarray类型。

### [test.py](./test.py)

将整个Pascal VOC 2012验证集中的图片跑一遍预测，通过真实的mask与预测的mask做最大匹配，得到一一对应的mask，并求出相关的指标，最后保存mask成图像，相关指标数据保存到Excel当中。

---


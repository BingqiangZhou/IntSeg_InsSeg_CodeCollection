# RSIS跑通过程记录

在作者提供的代码([Github地址](https://github.com/imatge-upc/rsis))的基础上，进行修改，实现推理预测。

## 创建环境以及安装相关Python包

**注意：首先得安装好CUDA、cuDNN以及GCC，[下载地址](../../README.md#实验环境)**

````bash

# 下载源代码
git clone https://github.com/imatge-upc/rsis
mv rsis RSIS
cd RSIS

# 创建虚拟环境
conda create -n rsis python=3.6
conda activate rsis

# 安装pytorch、torchvision
conda install pytorch torchvision cudatoolkit=11.1 -c pytorch -c nvidia

# 下载依赖相关包
pip install matplotlib pandas opencv-python tqdm openpyxl scipy

````

## 修改及编写代码
### [代码打包下载](https://github.com/BingqiangZhou/IntSeg_InsSeg_CodeCollection/releases/tag/rsis)

### 下载模型文件

**需要科学上网** [源Github库](https://github.com/imatge-upc/rsis)下提供了预训练模型的[下载地址](https://github.com/imatge-upc/rsis#pretrained-models)，我将`rsis-pascal`放在了本仓库的[Releaes](https://github.com/BingqiangZhou/IntSeg_InsSeg_CodeCollection/releases/tag/rsis)中，提供下载。

这里以`rsis-pascal`为例，下载后解压后放到`RSIS\models`文件夹中。

### 修改源码

1. 修改代码，解决如下两个找不到包的问题，在前面加一点即可，见[`src/modules/model.py`](./src/modules/model.py)。
```
Traceback (most recent call last):
  File "net.py", line 14, in <module>
    from modules.model import RSIS, FeatureExtractor
  File "./src\modules\model.py", line 3, in <module>
    from clstm import ConvLSTMCell
ModuleNotFoundError: No module named 'clstm'

Traceback (most recent call last):
  File "net.py", line 14, in <module>
    from modules.model import RSIS, FeatureExtractor
  File "./src\modules\model.py", line 10, in <module>
    from vision import VGG16, ResNet34, ResNet50, ResNet101
ModuleNotFoundError: No module named 'vision'
``` 

2. 修改代码，解决如下问题（要传入整形，但是传入了浮点型），在除法的位置加上一个`/`，让其整除即可，见[`src/modules/model.py`](https://github.com/BingqiangZhou/IntSeg_InsSeg_CodeCollection/blob/a783d18dc8c5b8a2a341abeeae6103dc018b60fe/ImageInstanceSegmentation/RSIS/src/modules/model.py#L43)。

``` 
Traceback (most recent call last):
  File "net.py", line 147, in <module>
    out_masks, out_classes, out_stops, spend_time = RIISNet(use_gpu=False, maxseqlen=maxseqlen).predict(x)
  File "net.py", line 69, in __init__
    self.encoder = FeatureExtractor(load_args)
  File "./src\modules\model.py", line 45, in __init__
    self.sk3 = nn.Conv2d(skip_dims_in[2],self.hidden_size/2,self.kernel_size,padding=self.padding)
  File "D:\Miniconda3\envs\rsis\lib\site-packages\torch\nn\modules\conv.py", line 388, in __init__
    False, _pair(0), groups, bias, padding_mode)
  File "D:\Miniconda3\envs\rsis\lib\site-packages\torch\nn\modules\conv.py", line 107, in __init__
    out_channels, in_channels // groups, *kernel_size))
TypeError: new() received an invalid combination of arguments - got (float, int, int, int), but expected one of:
 * (*, torch.device device)
 * (torch.Storage storage)
 * (Tensor other)
 * (tuple of ints size, *, torch.device device)
 * (object data, *, torch.device device)
```

### net.py

**1. 加载网络模型**
通过对[src/eval.py](./src/eval.py)进行取舍，去掉那部分不需要的代码，加入调用了的方法(做了一些小修改)，得以加载网络模型并进行预测。

**1. 输入图像预处理**

对图像做归一化操作（mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]），并转成[1, 3, h, w]大小的张量

**2. 后处理**

网络的输出结果主要包括：输出的mask, 输出的类别class, 输出的分数（可用于停止预测）。
后处理为将mask二值化，将所有输出转为ndarray类型。

### test.py

将整个Pascal VOC 2012验证集中的图片跑一遍预测，通过真实的mask与预测的mask做最大匹配，得到一一对应的mask，并求出相关的指标，最后保存mask成图像，相关指标数据保存到Excel当中。

---
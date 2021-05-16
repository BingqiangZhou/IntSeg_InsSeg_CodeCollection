# RPEIG跑通过程记录

这份代码有点特殊，([Github地址](https://github.com/aimerykong/Recurrent-Pixel-Embedding-for-Instance-Grouping))，没有给出从Embedding特征图到实例分割的到mask的相关代码，而是直接将其从高维随机映射到三维，做可视化。

这里由于Matlab的代码，我不太会，所以直接简单的去掉了`demo4_InstSegTraining_VOC2012/main000_metaStep1_visual_withoutMShift.m`的一些可以不需要的可视化，随后将特征图保存下来。[demo4_InstSegTraining_VOC2012/test.m](./demo4_InstSegTraining_VOC2012/test.m)是修改后的代码。

## 配置环境

这里需要编译安装MatConvnet，[参考: Matlab编译安装MatConvnet流程及问题解决](https://zhuanlan.zhihu.com/p/138587666)

## 修改及编写代码
### [代码打包下载](https://github.com/BingqiangZhou/IntSeg_InsSeg_CodeCollection/releases/tag/rpeig)

### 下载模型文件

**需要科学上网** [源Github库](https://github.com/aimerykong/Recurrent-Pixel-Embedding-for-Instance-Grouping)下提供了预训练模型的，我将作者给出的预训练模型`pairMMAbsReg_net-epoch-1.mat`放在了本仓库的[Releaes](https://github.com/BingqiangZhou/IntSeg_InsSeg_CodeCollection/releases/tag/rpeig)中，提供下载。

### test.py

这份代码，没有将网络摘出来，只是将[demo4_InstSegTraining_VOC2012/test.m](./demo4_InstSegTraining_VOC2012/test.m)保存下来的Embed ing特征图，用python语言加载，然后聚类，得到预测的mask，最后通过真实的mask与预测的mask做最大匹配，得到一一对应的mask，并求出相关的指标，最后保存mask成图像，相关指标数据保存到Excel当中。

---
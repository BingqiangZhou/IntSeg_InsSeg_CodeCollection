# 交互式图像分割

这里记录查看过的一些交互式图像代码，包括Github库、论文地址以及所作的一些尝试。

## 代码状态

* ✅  通过pytorch实现推理过程
* ✳️  通过OpenCV DNN模块读取模型以及其他方式（非pytorch）实现推理过程
* ⏹  有代码，没有模型，没有做
* ✖️  没有找到开源代码
* ❌  存在某种困难或者其他原因，战略性放弃
* 🔶  暂时未尝试

## 代码分类

这里主要按年份分类：

[2016](#2016), [2017](#2017), [2018](#2018), [2019](#2019), [2020](#2020), [2021](#2021)

### 2016

✖️ **Deep interactive object selection**
* [Paper PDF - https://arxiv.org/pdf/1603.04042](https://arxiv.org/pdf/1603.04042)
* 没有找到开源代码

### 2017

✖️ **Regional interactive image segmentation networks**
* [Paper PDF - https://ieeexplore.ieee.org/document/8237559](https://ieeexplore.ieee.org/document/8237559)
* 没有找到开源代码

✅ **Deep GrabCut for Object Selection**
* [源Github - https://github.com/jfzhang95/DeepGrabCut-PyTorch](https://github.com/jfzhang95/DeepGrabCut-PyTorch)
* [Paper PDF - https://arxiv.org/pdf/1711.09081](https://arxiv.org/pdf/1707.00243)
* 有提供模型文件
* [代码打包下载](https://github.com/BingqiangZhou/IntSeg_InsSeg_CodeCollection/releases/tag/deepgrabcut)
* [README](https://github.com/BingqiangZhou/IntSeg_InsSeg_CodeCollection/tree/master/InteractiveImageSegmentation/DeepGrabCut)

✅ **Deep extreme cut: From extreme points to object segmentation**
* [项目主页 - https://cvlsegmentation.github.io/dextr/](https://cvlsegmentation.github.io/dextr/)
* [源Github - https://github.com/scaelles/DEXTR-PyTorch/](https://github.com/scaelles/DEXTR-PyTorch/)
* [Paper PDF - https://arxiv.org/pdf/1711.09081](https://arxiv.org/pdf/1711.09081)
* 有提供模型文件
* [代码打包下载](https://github.com/BingqiangZhou/IntSeg_InsSeg_CodeCollection/releases/tag/dextr)
* [README](https://github.com/BingqiangZhou/IntSeg_InsSeg_CodeCollection/tree/master/InteractiveImageSegmentation/DEXTR)

🔶 **Annotating object instances with a polygon-RNN**
* [项目主页 - http://www.cs.toronto.edu/polyrnn/](http://www.cs.toronto.edu/polyrnn/)
* [Paper PDF - https://arxiv.org/pdf/1704.05548](https://arxiv.org/pdf/1704.05548)
* 项目已经改进更新为[polygon-RNN++](#polyrnn-pp)

### 2018

**Iteratively trained interactive segmentation**
* [源Github - https://github.com/sabarim/itis](https://github.com/sabarim/itis)
* [Paper PDF - https://arxiv.org/pdf/1805.04398](https://arxiv.org/pdf/1805.04398)

**SeedNet: Automatic seed generation with deep reinforcement learning for robust interactive segmentation**
* [源Github - https://github.com/kelawaad/SeedNet](https://github.com/kelawaad/SeedNet)
* [Paper PDF - https://openaccess.thecvf.com/content_cvpr_2018/papers/...pdf](https://openaccess.thecvf.com/content_cvpr_2018/papers/Song_SeedNet_Automatic_Seed_CVPR_2018_paper.pdf)

**Interactive image segmentation with latent diversity**
* [源Github - https://github.com/intel-isl/Intseg](https://github.com/intel-isl/Intseg)
* [Paper PDF - https://openaccess.thecvf.com/content_cvpr_2018/papers/...pdf](https://openaccess.thecvf.com/content_cvpr_2018/papers/Li_Interactive_Image_Segmentation_CVPR_2018_paper.pdf)

**A fully convolutional two-stream fusion network for interactive image segmentation**
* [源Github - https://github.com/cyh4/FCTSFN](https://github.com/cyh4/FCTSFN)
* [Paper PDF - https://arxiv.org/pdf/1807.02480](https://arxiv.org/pdf/1807.02480)

🔶 **Efficient interactive annotation of segmentation datasets with polygon-RNN++** <span id="polyrnn-pp"></span>
* [项目主页 - http://www.cs.toronto.edu/polyrnn/](http://www.cs.toronto.edu/polyrnn/)
* [源Github - https://github.com/fidler-lab/polyrnn-pp-pytorch](https://github.com/fidler-lab/polyrnn-pp-pytorch)
* [Paper PDF - https://arxiv.org/pdf/1803.09693](https://arxiv.org/pdf/1803.09693)

🔶 **Few-Shot Segmentation Propagation with Guided Networks**
* [源Github - https://github.com/shelhamer/revolver](https://github.com/shelhamer/revolver)
* [Paper PDF - https://arxiv.org/pdf/1806.07373](https://arxiv.org/pdf/1806.07373)


### 2019

**Interactive image segmentation via backpropagating refinement scheme**
* [源Github - https://github.com/wdjang/BRS-Interactive_segmentation](https://github.com/wdjang/BRS-Interactive_segmentation)
* [Paper PDF - https://openaccess.thecvf.com/content_CVPR_2019/papers/...pdf](https://openaccess.thecvf.com/content_CVPR_2019/papers/Jang_Interactive_Image_Segmentation_via_Backpropagating_Refinement_Scheme_CVPR_2019_paper.pdf)

✖️ **Interactive full image segmentation by considering all regions jointly**
* [Paper PDF - https://openaccess.thecvf.com/content_CVPR_2019/papers/...pdf](https://openaccess.thecvf.com/content_CVPR_2019/papers/Agustsson_Interactive_Full_Image_Segmentation_by_Considering_All_Regions_Jointly_CVPR_2019_paper.pdf)
* 没有找到开源代码

✖️ **MultiSeg: Semantically meaningful, scale-diverse segmentations from minimal user input**
* [Paper PDF - https://openaccess.thecvf.com/content_CVPR_2019/papers/...pdf](https://openaccess.thecvf.com/content_ICCV_2019/papers/Liew_MultiSeg_Semantically_Meaningful_Scale-Diverse_Segmentations_From_Minimal_User_Input_ICCV_2019_paper.pdf)
* 没有找到开源代码

🔶 **Fast interactive object annotation with curve-GCN**
* [源Github - https://github.com/fidler-lab/curve-gcn](https://github.com/fidler-lab/curve-gcn)
* [Paper PDF - https://openaccess.thecvf.com/content_CVPR_2019/papers/...pdf](https://openaccess.thecvf.com/content_CVPR_2019/papers/Ling_Fast_Interactive_Object_Annotation_With_Curve-GCN_CVPR_2019_paper.pdf)

🔶 **Object instance annotation with deep extreme level set evolution**
* [项目主页 - http://www.cs.toronto.edu/~zianwang/DELSE/delse.html](http://www.cs.toronto.edu/~zianwang/DELSE/delse.html)
* [源Github - https://github.com/fidler-lab/delse](https://github.com/fidler-lab/delse)
* [Paper PDF - https://openaccess.thecvf.com/content_CVPR_2019/papers/...pdf](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Object_Instance_Annotation_With_Deep_Extreme_Level_Set_Evolution_CVPR_2019_paper.pdf)

### 2020

**Interactive image segmentation with first click attention**
* [项目主页 - https://www.lin-zheng.com/fclick/](https://www.lin-zheng.com/fclick/)
* [源Github - https://github.com/frazerlin/fcanet](https://github.com/frazerlin/fcanet)
* [Paper PDF - https://openaccess.thecvf.com/content_CVPR_2019/papers/...pdf](https://openaccess.thecvf.com/content_CVPR_2020/papers/Lin_Interactive_Image_Segmentation_With_First_Click_Attention_CVPR_2020_paper.pdf)

**DISIR: Deep image segmentation with interactive refinement**
* [源Github - https://github.com/delair-ai/DISIR](https://github.com/delair-ai/DISIR)
* [Paper PDF - https://arxiv.org/pdf/2003.14200](https://arxiv.org/pdf/2003.14200)

**F-BRS: Rethinking backpropagating refinement for interactive segmentation**
* [官方Github - https://github.com/saic-vul/fbrs_interactive_segmentation](https://github.com/saic-vul/fbrs_interactive_segmentation)
* [修改版Github - https://github.com/jpconnel/fbrs-segmentation](https://github.com/jpconnel/fbrs-segmentation)
* [Paper PDF - https://openaccess.thecvf.com/content_CVPR_2020/papers/...pdf](https://openaccess.thecvf.com/content_CVPR_2020/papers/Sofiiuk_F-BRS_Rethinking_Backpropagating_Refinement_for_Interactive_Segmentation_CVPR_2020_paper.pdf)

**Getting to 99% accuracy in interactive segmentation**
* [源Github - https://github.com/MarcoForte/DeepInteractiveSegmentation](https://github.com/MarcoForte/DeepInteractiveSegmentation)
* [Paper PDF - https://arxiv.org/pdf/2003.07932](https://arxiv.org/pdf/2003.07932)

**Interactive Object Segmentation with Inside-Outside Guidance**
* [源Github - https://github.com/shiyinzhang/Inside-Outside-Guidance](https://github.com/shiyinzhang/Inside-Outside-Guidance)
* [Paper PDF - https://openaccess.thecvf.com/content_CVPR_2020/papers/...pdf](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhang_Interactive_Object_Segmentation_With_Inside-Outside_Guidance_CVPR_2020_paper.pdf)

### 2021

**Reviving Iterative Training with Mask Guidance for Interactive Segmentation**
* [源Github - https://github.com/saic-vul/ritm_interactive_segmentation](https://github.com/saic-vul/ritm_interactive_segmentation)
* [Paper PDF - https://arxiv.org/pdf/2102.06583](https://arxiv.org/pdf/2102.06583)

import os 

import time
import torch
import torchvision
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

# reference document: https://pytorch.org/vision/stable/models.html#mask-r-cnn
class MaskRCNN:
    def __init__(self, use_gpu=True, threshold=0.5) -> None:
        self.use_gpu =use_gpu
        self.threshold = threshold

        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True, 
                                                           progress=True, num_classes=91, 
                                                           pretrained_backbone=True, trainable_backbone_layers=None)
        self.model.eval()
        if use_gpu:
            self.model.cuda()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # 不需要归一化
        ])
        
    def __pre_process(self, x):
        x = self.transform(x)
        x = x.unsqueeze(dim=0) # (1 ,3, h, w)
        return x

    def __post_process(self, out):
        masks = out['masks'].cpu().numpy()
        masks[masks > self.threshold] = 1
        masks[masks <= self.threshold] = 0
        masks = np.squeeze(masks, axis=1)
        return masks, out['scores'].cpu().numpy(), out['boxes'].cpu().numpy(), out['labels'].cpu().numpy()

    def predict(self, x):
        x = self.__pre_process(x)
        if self.use_gpu:
            x = x.cuda()
        with torch.no_grad():
            start_time = time.time()
            # out = self.model([x])[0] # 这一种也可以，此时x对应大小为[3, h, w]
            out = self.model(x)[0] # 对应x为[1, 3, h, w]
            end_time = time.time()

        masks, scores, boxes, labels = self.__post_process(out)

        return masks, scores, boxes, labels, end_time-start_time

# model = MaskRCNN()

# image_path = r'E:\Datasets\iis_datasets\VOCdevkit\VOC2012\JPEGImages\2007_000033.jpg'
# num_objects = 3
# image = np.array(Image.open(image_path))
# plt.subplot(1, num_objects+1, 1)
# plt.imshow(image)

# masks, _, _, _, use_time = model.predict(image)
# for i in range(num_objects):
#     plt.subplot(1, num_objects+1, i+2)
#     mask = masks[i]
#     plt.imshow(mask)
# plt.show()
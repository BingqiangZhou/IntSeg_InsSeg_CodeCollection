import torch
import cv2 as cv
import time
from PIL import Image
import numpy as np
from model.fcanet import FCANet as Net
from core import Resize, CatPointMask, ToTensor
from scipy.ndimage.morphology import distance_transform_edt


class FCANet():
    def __init__(self, backbone='resnet', device=0, 
                    pretrained_file='./pretrained_model/fcanet-resnet.pth', record_time=True):
        print('Backbone is {}'.format(backbone))
        self.record_time = record_time
        self.model = Net(backbone=backbone)
        device_str = f"cuda:{device}" if device >= 0 else "cpu"
        self.device = torch.device(device_str)
        self.model = self.model.to(self.device)
        self.model.eval()
        state_dict=torch.load(pretrained_file,map_location=self.device)
        self.model.load_state_dict(state_dict)
        print('load from [{}]!'.format(pretrained_file))
    
    def predict(self, image, pos_points_mask, neg_points_mask, first_point_mask):
        h,w,_ = image.shape
        sample={}
        sample['img'] = image.copy()
        sample['gt'] = (np.ones((h,w))*255).astype(np.uint8)
        sample['pos_points_mask'] = pos_points_mask
        sample['neg_points_mask'] = neg_points_mask
        sample['first_point_mask'] = first_point_mask
        sample['pos_mask_dist_first'] = np.minimum(distance_transform_edt(1-sample['first_point_mask']), 255.0)*255.0
        Resize((int(w*512/min(h, w)),int(h*512/min(h, w))))(sample)
        CatPointMask(mode='DISTANCE_POINT_MASK_SRC', if_repair=False)(sample)
        ToTensor()(sample)
        input=[sample['img'].unsqueeze(0),  sample['pos_mask_dist_src'].unsqueeze(0), sample['neg_mask_dist_src'].unsqueeze(0), sample['pos_mask_dist_first'].unsqueeze(0)]
        for i in range(len(input)):
            input[i]=input[i].to(self.device)
        with torch.no_grad(): 
            start_time = time.time()
            output = self.model(input)
            end_time = time.time()
        result = torch.sigmoid(output.data.cpu()).numpy()[0,0,:,:]
        result = cv.resize(result, (w,h), interpolation=cv.INTER_LINEAR)     
        pred = (result>0.5).astype(np.uint8)
        if self.record_time:
            return pred, end_time-start_time
        return pred

# image = np.array(Image.open("./test.jpg"))

# pos_points_mask = np.zeros(image.shape[:2])
# neg_points_mask = np.zeros(image.shape[:2])
# first_point_mask = np.zeros(image.shape[:2])
# pos_points_mask[300, 150] = 1
# first_point_mask[300, 150] = 1

# net = FCANet(device=0)
# result, time = net.predict(image, pos_points_mask, neg_points_mask, first_point_mask)
# cv.imshow("out", result * 255)
# cv.waitKey(0)

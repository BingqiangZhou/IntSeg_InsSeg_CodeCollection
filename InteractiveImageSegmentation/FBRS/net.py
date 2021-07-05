import torch
import numpy as np
import time
from PIL import Image
import cv2 as cv
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from isegm.utils import exp
from isegm.inference import utils

class FBRS:
    def __init__(self, num_device=0, checkpoint='resnet34_dh128_sbd.pth', cfg_file='config.yml') -> None:
        
        device_str = "cpu" if num_device < 0 else f"cuda:{num_device}"
        self.device = torch.device(device_str)

        norm_radius = 260
        # limit_longest_size = 800

        torch.backends.cudnn.deterministic = True
        cfg = exp.load_config_file(cfg_file, return_edict=True)
        checkpoint_path = utils.find_checkpoint(cfg.INTERACTIVE_MODELS_PATH, checkpoint)
        self.model = utils.load_is_model(checkpoint_path, self.device, cpu_dist_maps=True, norm_radius=norm_radius)
        self.input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def predict(self, image_np, points):
        """
            image: RGB image, np.ndarray
            points: set of interactives, (y (h), x(x))
                list, such as: [[(200, 240), (180, 240), (220, 240), (-1, -1), (240, 200)]]
                before (-1, -1) is fg point, after (-1, -1) is bg point
                https://github.com/saic-vul/fbrs_interactive_segmentation/blob/a2f62973af21cb37f140135b1d743d8a7c498375/isegm/inference/predictors/base.py#L74
        """
        image = self.input_transform(image_np).unsqueeze(dim=0).to(self.device) # [3, h, w]

        inter = torch.tensor(points).to(self.device)

        out = self.model(image, inter)['instances'][0][0].cpu().detach().numpy() # [1, 1, h, w] -> [h, w], 是在Sigmoid之前的结果，出现负数

        out = (out > 0).astype(np.uint8)

        return out

# # # image_path = './images/2011_003271.jpg'
# image_path = './images/2007_000027.jpg'
# image_np = np.array(Image.open(image_path))
# input_transform = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize([.485, .456, .406], [.229, .224, .225])
#         ])

# image = input_transform(image_np).unsqueeze(dim=0).cuda()
# # 参考：https://github.com/saic-vul/fbrs_interactive_segmentation/blob/a2f62973af21cb37f140135b1d743d8a7c498375/isegm/inference/predictors/base.py#L74
# pos_clicks = [[(200, 240), (180, 240), (220, 240), (-1, -1), (240, 200)]]
# cfg_file = 'config.yml'
# norm_radius = 260
# device = torch.device("cuda:0")
# # limit_longest_size = 800
# gpu = 0
# # checkpoint = 'resnet50_dh128_lvis.pth'
# checkpoint = 'resnet34_dh128_sbd.pth'
# inter = torch.tensor(pos_clicks).cuda()
# cfg = exp.load_config_file(cfg_file, return_edict=True)
# checkpoint_path = utils.find_checkpoint(cfg.INTERACTIVE_MODELS_PATH, checkpoint)
# model = utils.load_is_model(checkpoint_path, device, cpu_dist_maps=True, norm_radius=norm_radius)

# out = model(image, inter)
# out_mask = out['instances'][0][0].cpu().detach().numpy()
# print(out_mask, out['instances'].shape)
# out_mask[out_mask > 0] = 1
# out_mask[out_mask <= 0] = 0
# # plt.imsave("out.png", out_mask)


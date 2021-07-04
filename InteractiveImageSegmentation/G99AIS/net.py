
import numpy as np
import cv2
import torch
import time

from networks.transforms import trimap_transform, groupnorm_normalise_image
from networks.models import build_model
from interaction import remove_non_fg_connected

class G99AIS():
    def __init__(self, weights_path="./models/InterSegSynthFT.pth", num_device=0, threshold=0.8):
        device_str = "cpu" if num_device < 0 else f"cuda:{num_device}"
        self.device = torch.device(device_str)
        self.threshold = threshold
        class Args():
            use_mask_input = True
            use_usr_encoder = True
            weights = weights_path
            device = self.device

        args = Args()
        self.model = build_model(args)
        self.model.eval()

    def __np_to_torch__(self, x):
        return torch.from_numpy(x).permute(2, 0, 1)[None, :, :, :].float().to(self.device)

    def __scale_input__(self, x: np.ndarray, scale_type) -> np.ndarray:
        ''' Scales so that min side length is 352 and sides are divisible by 8'''
        h, w = x.shape[:2]
        h1 = int(np.ceil(h / 32) * 32)
        w1 = int(np.ceil(w / 32) * 32)
        x_scale = cv2.resize(x, (w1, h1), interpolation=scale_type)
        return x_scale

    def predict(self, image_np: np.ndarray, trimap_np: np.ndarray, alpha_old_np: np.ndarray = None) -> np.ndarray:
        ''' Predict segmentation
            Parameters:
            image_np -- the image in rgb format between 0 and 1. Dimensions: (h, w, 3)
            trimap_np -- two channel trimap/Click map, first background then foreground. Dimensions: (h, w, 2)
            Returns:
            alpha: alpha matte/non-binary segmentation image between 0 and 1. Dimensions: (h, w)
        '''
        # return trimap_np[:,:,1] + (1-np.sum(trimap_np,-1))/2
        if alpha_old_np is None:
            alpha_old_np = np.zeros(image_np.shape[:2])
        else:
            alpha_old_np = remove_non_fg_connected(alpha_old_np, trimap_np[:, :, 1])

        image_np = image_np / 255.0

        h, w = trimap_np.shape[:2]
        image_scale_np = self.__scale_input__(image_np, cv2.INTER_LANCZOS4)
        trimap_scale_np = self.__scale_input__(trimap_np, cv2.INTER_NEAREST)
        alpha_old_scale_np = self.__scale_input__(alpha_old_np, cv2.INTER_LANCZOS4)

        with torch.no_grad():

            image_torch = self.__np_to_torch__(image_scale_np)
            trimap_torch = self.__np_to_torch__(trimap_scale_np)
            alpha_old_torch = self.__np_to_torch__(alpha_old_scale_np[:, :, None])

            trimap_transformed_torch = self.__np_to_torch__(trimap_transform(trimap_scale_np))
            image_transformed_torch = groupnorm_normalise_image(image_torch.clone(), format='nchw')
            alpha = self.model(image_transformed_torch, trimap_transformed_torch, alpha_old_torch, trimap_torch)
            alpha = cv2.resize(alpha[0].cpu().numpy().transpose((1, 2, 0)), (w, h), cv2.INTER_LANCZOS4)
        
        alpha[trimap_np[:, :, 0] == 1] = 0
        alpha[trimap_np[:, :, 1] == 1] = 1

        alpha = remove_non_fg_connected(alpha, trimap_np[:, :, 1])

        alpha = (alpha > self.threshold).astype(np.uint8)

        return alpha


# from PIL import Image
# import cv2 as cv

# image = np.array(Image.open("./ims/images/21077.png"))
# fg = np.zeros(image.shape[:2])
# bg = np.zeros(image.shape[:2])
# net = G99AIS()

# fg[160, 240] = 1

# trimap = np.stack([bg, fg], axis=-1)
# out = net.predict(image, trimap)

# cv.imshow("out", out*255)
# cv.waitKey(0)
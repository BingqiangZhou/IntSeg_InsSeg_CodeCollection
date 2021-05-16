import time
import cv2 as cv
import matplotlib.pyplot as plt

from fcos_core.config import cfg
from demo.predictor import COCODemo

class EmbedMask:
    def __init__(self, use_gpu=True, threshold=0.2) -> None:
        # config_file = 'configs\embed_mask\embed_mask_R50_1x.yaml'
        # weights = 'models\embed_mask_R50_1x.pth'

        config_file = 'configs\embed_mask\embed_mask_R101_ms_3x.yaml'
        weights = 'models\embed_mask_R101_ms_3x.pth'

        cfg.merge_from_file(config_file)
        cfg.MODEL.WEIGHT = weights
        cfg.MODEL.DEVICE = 'cpu'
        if use_gpu:
            cfg.MODEL.DEVICE = 'cuda:0'

        cfg.freeze()

        self.coco_demo = COCODemo(cfg)

        self.threshold = threshold

    def predict(self, image_path):
        img = cv.imread(image_path)

        prediction, inference_time = self.coco_demo.compute_prediction(img)
        # print(prediction) # BoxList(num_boxes=50, image_width=500, image_height=366, mode=xyxy)
        pred_masks = prediction.get_field('mask')
        pred_mask_scores = prediction.get_field("scores")
        pred_class_labels = prediction.get_field("labels")
        pred_boxes = prediction.bbox
        # print(pred_masks, pred_mask_scores, pred_boxes, pred_class_labels)

        masks = []
        mask_scores = []
        class_labels = []
        bboxs = []
        for i, mask in enumerate(pred_masks):
            if pred_mask_scores[i] > self.threshold:
                masks.append(mask[0].numpy()) # (1, h, w)
                mask_scores.append(pred_mask_scores[i])
                class_labels.append(pred_class_labels[i])
                bboxs.append(pred_boxes[i])

        return masks, mask_scores, class_labels, bboxs, inference_time

image_path = r'E:\Datasets\iis_datasets\VOCdevkit\VOC2012\JPEGImages\2007_000033.jpg'

net = EmbedMask()
masks, mask_scores, class_labels, bboxs, inference_time = net.predict(image_path)
num_object = len(masks)
for i, mask in enumerate(masks):
    plt.subplot(1, num_object, i+1)
    plt.imshow(mask)
    plt.title(class_labels[i])
plt.show()

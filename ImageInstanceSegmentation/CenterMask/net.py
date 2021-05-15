import cv2 as cv
import matplotlib.pyplot as plt

from maskrcnn_benchmark.config import cfg
from demo.predictor import COCODemo

class CenterMask:
    def __init__(self, config_file = 'configs\centermask\centermask_R_50_FPN_ms_2x.yaml', 
                    weights = 'models\centermask-R-50-FPN-ms-2x.pth',
                    use_gpu=True, threshold=0.2) -> None:
        # config_file = 'configs\centermask\centermask_R_50_FPN_ms_2x.yaml'
        # weights = 'models\centermask-R-50-FPN-ms-2x.pth'
        conf_th = 0.5 # confidence_threshold

        cfg.merge_from_file(config_file)
        cfg.MODEL.WEIGHT = weights
        cfg.MODEL.DEVICE = 'cpu'
        if use_gpu:
            cfg.MODEL.DEVICE = 'cuda:0'

        cfg.freeze()

        self.coco_demo = COCODemo(
            cfg,
            confidence_threshold=conf_th,
            display_text = False,
            display_scores = False
        )

        self.threshold = threshold

    def predict(self, image_path):
        img = cv.imread(image_path)

        prediction, inference_time = self.coco_demo.compute_prediction(img)
        # print(prediction) # BoxList(num_boxes=50, image_width=500, image_height=366, mode=xyxy)
        pred_masks = prediction.get_field('mask') # [50, 1, h, w]
        pred_mask_scores = prediction.get_field("mask_scores") # [50]
        pred_class_labels = prediction.get_field("labels") # [50]
        pred_boxes = prediction.bbox # [50, 4]
        # print(pred_masks, pred_mask_scores, pred_boxes, pred_class_labels)
        # print(pred_masks.shape, pred_mask_scores.shape, pred_boxes.shape, pred_class_labels.shape)

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

# image_path = r'E:\Datasets\iis_datasets\VOCdevkit\VOC2012\JPEGImages\2007_000033.jpg'

# net = CenterMask()
# masks, mask_scores, class_labels, bboxs, inference_time = net.predict(image_path)
# num_object = len(masks)
# for i, mask in enumerate(masks):
#     plt.subplot(1, num_object, i+1)
#     plt.imshow(mask)
#     plt.title(class_labels[i])
# plt.show()



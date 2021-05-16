import matplotlib.pyplot as plt

from mmdet.apis import inference_detector, init_detector, show_result_pyplot


class RefineMask:
    def __init__(self, config_file = r'configs\refinemask\coco\r50-refinemask-2x.py',
                    checkpoint = r'models\r50-coco-2x.pth',
                    use_gpu=True, gpu=0, threshold=0.5) -> None:
        device = 'cpu'
        if use_gpu:
            device = 'cuda:' + str(gpu)

        self.threshold = threshold
        
        # build the model from a config file and a checkpoint file
        self.model = init_detector(config_file, checkpoint, device=device) # 修改了mmdet\models\builder.py中build_detector，不输出网络信息
    
    def predict(self, image_path):
        # test a single image
        
        result, inference_time = inference_detector(self.model, image_path) # 修改了mmdet\apis\inference.py，加入了计时
        bbox_results, mask_results = result
        # mask：80个分类，list[list],，每个分类下对应一个list
        # bbox的前四个值是bbox坐标，第五个值是分数
        
        # show the results
        # show_result_pyplot(model, img, result, score_thr=0.3)
        
        # 返回分数大于threshold的bbox以及mask
        bboxs = []
        masks = []
        classes = [] # 对应分类 https://www.jianshu.com/p/16b2e32d9edf, https://blog.csdn.net/weixin_41466947/article/details/98783700
        for i, bbox in enumerate(bbox_results):
            for j in range(bbox.shape[0]):
                if bbox[j][4] > self.threshold:
                    # print(f"class {i}, object {j}, score {bbox[j][4]}")
                    bboxs.append(bbox[j][:4])
                    masks.append(mask_results[i][j])
                    classes.append(i)
        return masks, bboxs, classes, inference_time

# image_path = r'E:\Datasets\iis_datasets\VOCdevkit\VOC2012\JPEGImages\2007_000033.jpg'

# net = RefineMask(threshold=0.8)
# masks, bboxs, classes, inference_time = net.predict(image_path)

# num_object = len(masks)
# for i, mask in enumerate(masks):
#     plt.subplot(1, num_object, i+1)
#     plt.imshow(mask)
#     plt.title(classes[i])
# plt.show()

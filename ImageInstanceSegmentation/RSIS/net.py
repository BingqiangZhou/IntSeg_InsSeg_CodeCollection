import os
import sys

sys.path.append("./src/")

import time
import torch
import pickle
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from modules.model import RSIS, FeatureExtractor

def load_checkpoint(model_name='rsis-pascal', use_gpu=True):
    models_dir = './models'
    if use_gpu:
        encoder_dict = torch.load(os.path.join(models_dir, model_name, 'encoder.pt'))
        decoder_dict = torch.load(os.path.join(models_dir, model_name, 'decoder.pt'))
        enc_opt_dict = torch.load(os.path.join(models_dir, model_name, 'enc_opt.pt'))
        dec_opt_dict = torch.load(os.path.join(models_dir, model_name, 'dec_opt.pt'))
    else:
        encoder_dict = torch.load(os.path.join(models_dir, model_name, 'encoder.pt'), map_location=lambda storage, location: storage)
        decoder_dict = torch.load(os.path.join(models_dir, model_name, 'decoder.pt'), map_location=lambda storage, location: storage)
        enc_opt_dict = torch.load(os.path.join(models_dir, model_name, 'enc_opt.pt'), map_location=lambda storage, location: storage)
        dec_opt_dict = torch.load(os.path.join(models_dir, model_name, 'dec_opt.pt'), map_location=lambda storage, location: storage)
    # save parameters for future use
    args = pickle.load(open(os.path.join(models_dir, model_name,'args.pkl'), 'rb'))

    return encoder_dict, decoder_dict, enc_opt_dict, dec_opt_dict, args

def check_parallel(encoder_dict, decoder_dict):
	# check if the model was trained using multiple gpus
    trained_parallel = False
    for k, v in encoder_dict.items():
        if k[:7] == "module.":
            trained_parallel = True
        break
    if trained_parallel:
        # create new OrderedDict that does not contain "module."
        new_encoder_state_dict = {} # OrderedDict()
        new_decoder_state_dict = {} # OrderedDict()
        for k, v in encoder_dict.items():
            name = k[7:]  # remove "module."
            new_encoder_state_dict[name] = v
        for k, v in decoder_dict.items():
            name = k[7:]  # remove "module."
            new_decoder_state_dict[name] = v
        encoder_dict = new_encoder_state_dict
        decoder_dict = new_decoder_state_dict

    return encoder_dict, decoder_dict 

class RIISNet:
    def __init__(self, use_gpu=True, maxseqlen=20, record_spend_time=True) -> None:
        self.image_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        ])

        encoder_dict, decoder_dict, _, _, load_args = load_checkpoint(use_gpu=use_gpu)
        load_args.use_gpu = use_gpu
        load_args.batch_size = 1
        load_args.maxseqlen = maxseqlen
        self.load_args = load_args
        # print(load_args)
        self.encoder = FeatureExtractor(load_args)
        self.decoder = RSIS(load_args)
        
        self.record_spend_time = record_spend_time

        encoder_dict, decoder_dict = check_parallel(encoder_dict,decoder_dict)
        self.encoder.load_state_dict(encoder_dict)
        self.decoder.load_state_dict(decoder_dict)

        if use_gpu:
            self.encoder.cuda()
            self.decoder.cuda()

        self.encoder.eval()
        self.decoder.eval()


    def predict(self, x, iter_times=0):
        if iter_times == 0:
            T = self.load_args.maxseqlen
        else:
            T = iter_times
        
        hidden = None
        out_masks = []
        out_classes = []
        out_stops = []
        
        x = self.__pre_process(x)

        if self.load_args.use_gpu:
            x = x.cuda()
        
        if self.record_spend_time:
            start_time = time.time()

        feats = self.encoder(x)
        # loop over sequence length and get predictions
        for t in range(0, T):
            out_mask, out_class, out_stop, hidden = self.decoder(feats, hidden)
            upsample_match = torch.nn.UpsamplingBilinear2d(size = (x.size()[-2],x.size()[-1]))
            out_mask = upsample_match(out_mask)
            # get predictions in list to concat later
            out_masks.append(out_mask)
            out_classes.append(out_class)
            out_stops.append(out_stop)
        
        if self.record_spend_time:
            end_time = time.time()
        
        out_masks = torch.cat(out_masks, 1)
        out_classes = torch.stack(out_classes, 0)
        out_stops = torch.stack(out_stops, 0)

        out_masks, out_classes, out_stops = self.__post_process(out_masks, out_classes, out_stops)

        if self.record_spend_time:
            return out_masks, out_classes, out_stops, (end_time - start_time) / T
        return out_masks, out_classes, out_stops

    def __pre_process(self, x):
        x = self.image_transforms(x)
        x = torch.unsqueeze(x, dim=0) # (3, h, w) -> (1, 3, h, w)
        return x.to(torch.float32)

    def __post_process(self, out_masks, out_classes, out_stops):
        out_masks, out_classes, out_stops = torch.sigmoid(out_masks).cpu().detach().numpy(), out_classes.cpu().detach().numpy(), torch.sigmoid(out_stops).cpu().detach().numpy()
        out_masks[out_masks > self.load_args.mask_th] = 1
        out_masks[out_masks <= self.load_args.mask_th] = 0 # self.load_args.mask_th = 0.5
        return out_masks[0], out_classes, out_stops

# maxseqlen = 20
# image_dir = r'E:\Datasets\iis_datasets\VOCdevkit\VOC2012\JPEGImages'
# image_name = '2007_000033'
# # image_name = '2007_000129'
# image_path = os.path.join(image_dir, image_name+'.jpg')
# x = Image.open(image_path)
# # x = np.random.rand(100, 100, 3)
# out_masks, out_classes, out_stops, spend_time = RIISNet(use_gpu=False, maxseqlen=maxseqlen).predict(x)
# # print(out_masks.shape, out_classes.shape, out_stops.shape, out_stops)

# CLASSES = ['<eos>','airplane', 'bicycle', 'bird', 'boat',
#             'bottle', 'bus', 'car', 'cat', 'chair',
#             'cow', 'dining table', 'dog', 'horse',
#             'motorcycle', 'person', 'potted plant',
#             'sheep', 'sofa', 'train', 'tv']

# for i in range(maxseqlen):
#     plt.subplot(2, 10, i+1)
#     plt.imshow(out_masks[i])
#     plt.title(CLASSES[np.argmax(out_classes[i])])
# plt.show()
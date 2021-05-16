import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

import numpy as np
from mmcv.cnn import ConvModule, build_upsample_layer
from mmcv.ops.roi_align import roi_align

from mmdet.core.mask.structures import polygon_to_bitmap, BitmapMasks
from mmdet.models.builder import HEADS, build_loss, build_roi_extractor
from .fcn_mask_head import _do_paste_mask
from .fcn_mask_head import BYTES_PER_FLOAT, GPU_MEM_LIMIT


class MultiBranchFusion(nn.Module):

    def __init__(self, feat_dim, dilations=[1, 3, 5]):
        super(MultiBranchFusion, self).__init__()

        for idx, dilation in enumerate(dilations):
            self.add_module(f'dilation_conv_{idx + 1}', ConvModule(
                feat_dim, feat_dim, kernel_size=3, padding=dilation, dilation=dilation))

        self.merge_conv = ConvModule(feat_dim, feat_dim, kernel_size=1, act_cfg=None)

    def forward(self, x):
        feat_1 = self.dilation_conv_1(x)
        feat_2 = self.dilation_conv_2(x)
        feat_3 = self.dilation_conv_3(x)
        out_feat = self.merge_conv(feat_1 + feat_2 + feat_3)
        return out_feat


class SFMStage(nn.Module):

    def __init__(self,
                 semantic_in_channel=256,
                 semantic_out_channel=256,
                 instance_in_channel=256,
                 instance_out_channel=256,
                 dilations=[1, 3, 5],
                 out_size=14,
                 num_classes=80,
                 semantic_out_stride=4,
                 mask_use_sigmoid=False,
                 upsample_cfg=dict(type='bilinear', scale_factor=2)):
        super(SFMStage, self).__init__()

        self.semantic_out_stride = semantic_out_stride
        self.mask_use_sigmoid = mask_use_sigmoid
        self.num_classes = num_classes

        # for extracting instance-wise semantic feats
        self.semantic_transform_in = nn.Conv2d(semantic_in_channel, semantic_out_channel, 1)
        self.semantic_roi_extractor = build_roi_extractor(dict(
                type='SingleRoIExtractor',
                roi_layer=dict(type='RoIAlign', output_size=out_size, sampling_ratio=0),
                out_channels=semantic_out_channel,
                featmap_strides=[semantic_out_stride, ]))
        self.semantic_transform_out = nn.Conv2d(semantic_out_channel, semantic_out_channel, 1)

        self.instance_logits = nn.Conv2d(instance_in_channel, num_classes, 1)

        fuse_in_channel = instance_in_channel + semantic_out_channel + 2
        self.fuse_conv = nn.ModuleList([
            nn.Conv2d(fuse_in_channel, instance_in_channel, 1),
            MultiBranchFusion(instance_in_channel, dilations=dilations)])

        self.fuse_transform_out = nn.Conv2d(instance_in_channel, instance_out_channel - 2, 1)
        self.upsample = build_upsample_layer(upsample_cfg.copy())
        self.relu = nn.ReLU(inplace=True)

        self._init_weights()

    def _init_weights(self):
        for m in [self.semantic_transform_in, self.semantic_transform_out, self.instance_logits, self.fuse_transform_out]:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)

        for m in self.fuse_conv:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, instance_feats, semantic_feat, semantic_pred, rois, roi_labels):
        concat_tensors = [instance_feats]

        # instance-wise semantic feats
        semantic_feat = self.relu(self.semantic_transform_in(semantic_feat))
        ins_semantic_feats = self.semantic_roi_extractor([semantic_feat,], rois)
        ins_semantic_feats = self.relu(self.semantic_transform_out(ins_semantic_feats))
        concat_tensors.append(ins_semantic_feats)

        # instance masks
        instance_preds = self.instance_logits(instance_feats)[torch.arange(len(rois)), roi_labels][:, None]
        _instance_preds = instance_preds.sigmoid() if self.mask_use_sigmoid else instance_preds
        instance_masks = F.interpolate(_instance_preds, instance_feats.shape[-2], mode='bilinear', align_corners=True)
        concat_tensors.append(instance_masks)

        # instance-wise semantic masks
        fake_rois = rois.clone()
        fake_rois[:, 0] = torch.zeros(len(rois))
        _semantic_pred = semantic_pred.sigmoid() if self.mask_use_sigmoid else semantic_pred
        ins_semantic_masks = roi_align(
            _semantic_pred, fake_rois, instance_feats.shape[-2:], 1.0 / self.semantic_out_stride, 0, 'avg', True)
        ins_semantic_masks = F.interpolate(
            ins_semantic_masks, instance_feats.shape[-2:], mode='bilinear', align_corners=True)
        concat_tensors.append(ins_semantic_masks)

        # fuse instance feats & instance masks & semantic feats & semantic masks
        fused_feats = torch.cat(concat_tensors, dim=1)
        for conv in self.fuse_conv:
            fused_feats = self.relu(conv(fused_feats))

        fused_feats = self.relu(self.fuse_transform_out(fused_feats))
        fused_feats = self.relu(self.upsample(fused_feats))

        # concat instance and semantic masks with fused feats again
        instance_masks = F.interpolate(_instance_preds, fused_feats.shape[-2], mode='bilinear', align_corners=True)
        ins_semantic_masks = F.interpolate(ins_semantic_masks, fused_feats.shape[-2], mode='bilinear', align_corners=True)
        fused_feats = torch.cat([fused_feats, instance_masks, ins_semantic_masks], dim=1)

        return instance_preds, fused_feats


@HEADS.register_module()
class RefineMaskHead(nn.Module):

    def __init__(self,
                 num_convs_instance=2,
                 num_convs_semantic=4,
                 conv_in_channels_instance=256,
                 conv_in_channels_semantic=256,
                 conv_kernel_size_instance=3,
                 conv_kernel_size_semantic=3,
                 conv_out_channels_instance=256,
                 conv_out_channels_semantic=256,
                 conv_cfg=None,
                 norm_cfg=None,
                 dilations=[1, 3, 5],
                 semantic_out_stride=4,
                 mask_use_sigmoid=False,
                 stage_num_classes=[80, 80, 80, 80],
                 stage_sup_size=[14, 28, 56, 112],
                 upsample_cfg=dict(type='bilinear', scale_factor=2),
                 loss_cfg=dict(
                    type='RefineCrossEntropyLoss',
                    stage_instance_loss_weight=[0.25, 0.5, 0.75, 1.0],
                    semantic_loss_weight=1.0,
                    boundary_width=2,
                    start_stage=1)):
        super(RefineMaskHead, self).__init__()

        self.num_convs_instance = num_convs_instance
        self.conv_kernel_size_instance = conv_kernel_size_instance
        self.conv_in_channels_instance = conv_in_channels_instance
        self.conv_out_channels_instance = conv_out_channels_instance

        self.num_convs_semantic = num_convs_semantic
        self.conv_kernel_size_semantic = conv_kernel_size_semantic
        self.conv_in_channels_semantic = conv_in_channels_semantic
        self.conv_out_channels_semantic = conv_out_channels_semantic

        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.semantic_out_stride = semantic_out_stride
        self.stage_sup_size = stage_sup_size
        self.stage_num_classes = stage_num_classes

        self._build_conv_layer('instance')
        self._build_conv_layer('semantic')
        self.loss_func = build_loss(loss_cfg)

        assert len(self.stage_sup_size) > 1
        self.stages = nn.ModuleList()
        out_channel = conv_out_channels_instance
        for idx, out_size in enumerate(self.stage_sup_size[:-1]):
            in_channel = out_channel
            out_channel = in_channel // 2

            new_stage = SFMStage(
                semantic_in_channel=conv_out_channels_semantic,
                semantic_out_channel=in_channel,
                instance_in_channel=in_channel,
                instance_out_channel=out_channel,
                dilations=dilations,
                out_size=out_size,
                num_classes=self.stage_num_classes[idx],
                semantic_out_stride=semantic_out_stride,
                mask_use_sigmoid=mask_use_sigmoid,
                upsample_cfg=upsample_cfg)

            self.stages.append(new_stage)

        self.final_instance_logits = nn.Conv2d(out_channel, self.stage_num_classes[-1], 1)
        self.semantic_logits = nn.Conv2d(conv_out_channels_semantic, 1, 1)
        self.relu = nn.ReLU(inplace=True)

    def _build_conv_layer(self, name):
        out_channels = getattr(self, f'conv_out_channels_{name}')
        conv_kernel_size = getattr(self, f'conv_kernel_size_{name}')

        convs = []
        for i in range(getattr(self, f'num_convs_{name}')):
            in_channels = getattr(self, f'conv_in_channels_{name}') if i == 0 else out_channels
            conv = ConvModule(in_channels, out_channels, conv_kernel_size, dilation=1, padding=1)
            convs.append(conv)

        self.add_module(f'{name}_convs', nn.ModuleList(convs))

    def init_weights(self):
        for m in [self.final_instance_logits, self.semantic_logits]:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)

    def forward(self, instance_feats, semantic_feat, rois, roi_labels):
        for conv in self.instance_convs:
            instance_feats = conv(instance_feats)

        for conv in self.semantic_convs:
            semantic_feat = conv(semantic_feat)

        semantic_pred = self.semantic_logits(semantic_feat)

        stage_instance_preds = []
        for stage in self.stages:
            instance_preds, instance_feats = stage(instance_feats, semantic_feat, semantic_pred, rois, roi_labels)
            stage_instance_preds.append(instance_preds)

        # for LVIS, use class-agnostic classifier for the last stage
        if self.stage_num_classes[-1] == 1:
            roi_labels = roi_labels.clamp(max=0)

        instance_preds = self.final_instance_logits(instance_feats)[torch.arange(len(rois)), roi_labels][:, None]
        stage_instance_preds.append(instance_preds)

        return stage_instance_preds, semantic_pred

    def get_targets(self, pos_bboxes, pos_assigned_gt_inds, gt_masks):

        def _generate_instance_targets(pos_proposals, pos_assigned_gt_inds, gt_masks, mask_size=None):
            device = pos_proposals.device
            proposals_np = pos_proposals.cpu().numpy()
            maxh, maxw = gt_masks.height, gt_masks.width
            proposals_np[:, [0, 2]] = np.clip(proposals_np[:, [0, 2]], 0, maxw)
            proposals_np[:, [1, 3]] = np.clip(proposals_np[:, [1, 3]], 0, maxh)
            pos_assigned_gt_inds = pos_assigned_gt_inds.cpu().numpy()

            # crop and resize the instance mask
            resize_masks = gt_masks.crop_and_resize(
                proposals_np, _pair(mask_size), inds=pos_assigned_gt_inds, device=device,).to_ndarray()
            instance_targets = torch.from_numpy(resize_masks).float().to(device)  # Tensor(Bitmaps)

            return instance_targets

        # multi-stage instance mask targets
        stage_instance_targets = [_generate_instance_targets(
            pos_bboxes, pos_assigned_gt_inds, gt_masks, mask_size=mask_size) for mask_size in self.stage_sup_size]

        # binary image semantic target
        if isinstance(gt_masks, BitmapMasks):
            instance_masks = torch.from_numpy(gt_masks.to_ndarray()).to(device=pos_bboxes.device, dtype=torch.float32)
        else:
            im_height, im_width = gt_masks.height, gt_masks.width
            instance_masks = [polygon_to_bitmap(polygon, im_height, im_width) for polygon in gt_masks]
            instance_masks = torch.from_numpy(np.stack(instance_masks)).to(device=pos_bboxes.device, dtype=torch.float32)
        semantic_target = instance_masks.max(dim=0, keepdim=True)[0]

        return stage_instance_targets, semantic_target

    def loss(self, stage_instance_preds, semantic_pred, stage_instance_targets, semantic_target):

        loss_instance, loss_semantic = self.loss_func(
            stage_instance_preds, semantic_pred, stage_instance_targets, semantic_target)

        return dict(loss_instance=loss_instance), dict(loss_semantic=loss_semantic)

    def get_seg_masks(self, mask_pred, det_bboxes, det_labels, rcnn_test_cfg, ori_shape, scale_factor, rescale):

        mask_pred = mask_pred.sigmoid()

        device = mask_pred[0].device
        bboxes = det_bboxes[:, :4]
        labels = det_labels

        if rescale:
            img_h, img_w = ori_shape[:2]
        else:
            img_h = np.round(ori_shape[0] * scale_factor).astype(np.int32)
            img_w = np.round(ori_shape[1] * scale_factor).astype(np.int32)
            scale_factor = 1.0

        if not isinstance(scale_factor, (float, torch.Tensor)):
            scale_factor = bboxes.new_tensor(scale_factor)
        bboxes = bboxes / scale_factor

        N = len(mask_pred)
        # The actual implementation split the input into chunks,
        # and paste them chunk by chunk.
        if device.type == 'cpu':
            # CPU is most efficient when they are pasted one by one with
            # skip_empty=True, so that it performs minimal number of
            # operations.
            num_chunks = N
        else:
            # GPU benefits from parallelism for larger chunks,
            # but may have memory issue
            num_chunks = int(
                np.ceil(N * img_h * img_w * BYTES_PER_FLOAT / GPU_MEM_LIMIT))
            assert (num_chunks <= N), 'Default GPU_MEM_LIMIT is too small; try increasing it'
        chunks = torch.chunk(torch.arange(N, device=device), num_chunks)

        threshold = rcnn_test_cfg.mask_thr_binary
        im_mask = torch.zeros(
            N,
            img_h,
            img_w,
            device=device,
            dtype=torch.bool if threshold >= 0 else torch.uint8)

        if mask_pred.shape[1] > 1:
            mask_pred = mask_pred[range(N), labels][:, None]

        for inds in chunks:
            masks_chunk, spatial_inds = _do_paste_mask(
                mask_pred[inds],
                bboxes[inds],
                img_h,
                img_w,
                skip_empty=device.type == 'cpu')

            if threshold >= 0:
                masks_chunk = (masks_chunk >= threshold).to(dtype=torch.bool)
            else:
                # for visualization and debugging
                masks_chunk = (masks_chunk * 255).to(dtype=torch.uint8)

            im_mask[(inds, ) + spatial_inds] = masks_chunk

        im_segms = [im_mask[i].cpu().numpy() for i in range(N)]
        return im_segms

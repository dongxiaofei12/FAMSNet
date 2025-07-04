# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from torch import Tensor
from typing import Tuple, Union

from mmcv.cnn import ConvModule, build_activation_layer, build_norm_layer
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.losses import accuracy
from mmseg.models.utils import resize
from mmseg.registry import MODELS
from mmseg.utils import OptConfigType, SampleList


@MODELS.register_module()
class GCNetHead(BaseDecodeHead):
    """Decode head for RDRNetV2.

    Args:
        in_channels (int): Number of input channels.
        channels (int): Number of output channels.
        num_classes (int): Number of classes.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict, optional): Config dict for activation layer.
            Default: dict(type='ReLU', inplace=True).
    """

    def __init__(self,
                 in_channels: int,
                 channels: int,
                 num_classes: int,
                 norm_cfg: OptConfigType = dict(type='BN', requires_grad=True),
                 act_cfg: OptConfigType = dict(type='ReLU', inplace=True),
                 **kwargs):
        super().__init__(
            in_channels,
            channels,
            num_classes=num_classes,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            **kwargs)

        self.head = self._make_base_head(self.in_channels, self.channels)

        self.aux_head_c4 = self._make_base_head(self.in_channels // 2, self.channels)
        self.aux_cls_seg_c4 = nn.Conv2d(self.channels, self.out_channels, kernel_size=1)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(
            self,
            inputs: Union[Tensor,
                          Tuple[Tensor]]) -> Union[Tensor, Tuple[Tensor]]:
        if self.training:
            c4_feat, c6_feat = inputs
            c4_feat = self.aux_head_c4(c4_feat)
            c4_feat = self.aux_cls_seg_c4(c4_feat)
            c6_feat = self.head(c6_feat)
            c6_feat = self.cls_seg(c6_feat)

            return c4_feat, c6_feat
        else:
            c6_feat = self.head(inputs)
            c6_feat = self.cls_seg(c6_feat)

            return c6_feat

    def _make_base_head(self, in_channels: int,
                        channels: int) -> nn.Sequential:
        layers = [
            build_norm_layer(self.norm_cfg, in_channels)[1],
            build_activation_layer(self.act_cfg),
            ConvModule(
                in_channels,
                channels,
                kernel_size=3,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                order=('conv', 'norm', 'act')),
        ]

        return nn.Sequential(*layers)

    def loss_by_feat(self, seg_logits: Tuple[Tensor],
                     batch_data_samples: SampleList) -> dict:
        loss = dict()
        c4_logit, c6_logit = seg_logits
        seg_label = self._stack_batch_gt(batch_data_samples)

        c4_logit = resize(
            c4_logit,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        c6_logit = resize(
            c6_logit,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        seg_label = seg_label.squeeze(1)

        loss['loss_c4'] = self.loss_decode[0](c4_logit, seg_label)
        loss['loss_c6'] = self.loss_decode[1](c6_logit, seg_label)
        loss['acc_seg'] = accuracy(
            c6_logit, seg_label, ignore_index=self.ignore_index)

        return loss
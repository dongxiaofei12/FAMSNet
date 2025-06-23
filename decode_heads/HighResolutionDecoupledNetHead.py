# custom_head.py
import torch
import torch.nn as nn

from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.registry import MODELS


@MODELS.register_module()
class HighResolutionDecoupledNetHead(BaseDecodeHead):
    def __init__(self, **kwargs):
        super(HighResolutionDecoupledNetHead, self).__init__(**kwargs)
        self.conv_seg = nn.Identity()


    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        x = self._transform_inputs(inputs)

        return x
    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)
        output = self.cls_seg(output)
        return output
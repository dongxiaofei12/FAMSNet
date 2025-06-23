from functools import partial

from torch import nn

from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.utils import build_norm_layer
from mmseg.registry import MODELS
from timm.models.layers import trunc_normal_

@MODELS.register_module()
class MlpHead(BaseDecodeHead):
    """ MLP classification head
    """

    def __init__(self, mlp_ratio=3, act_layer=nn.GELU,
                 norm_cfg=dict(type='LN2d', eps=1e-6), drop=0., bias=True, **kwargs):
        super().__init__(**kwargs)
        hidden_features = int(mlp_ratio * self.in_channels)
        self.fc1 = nn.Linear(self.in_channels, hidden_features, bias=bias)
        self.act = act_layer()
        self.norm = build_norm_layer(norm_cfg, hidden_features)
        self.fc2 = nn.Linear(hidden_features, self.num_classes, bias=bias)
        self.drop = nn.Dropout(drop)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

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
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.fc1(x)
        x = self.act(x)
        x = self.norm(x, data_format='channel_last')
        x = self.drop(x)
        x = self.fc2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        return x

    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)
        return output
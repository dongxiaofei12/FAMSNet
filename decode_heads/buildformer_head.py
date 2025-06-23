import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead

from mmseg.models.backbones.buildformer import ConvBNAct, Conv  # 确保导入必要的模块


@HEADS.register_module()
class BuildFormerHead(BaseDecodeHead):
    def __init__(self, decoder_channels=384, encoder_channels=(64, 128, 256, 512), num_classes=2, **kwargs):
        super(BuildFormerHead, self).__init__(num_classes=num_classes, **kwargs)

        # 定义自定义解码器
        self.conv_seg = nn.Sequential(
            ConvBNAct(decoder_channels, encoder_channels[0]),
            nn.Dropout(0.1),
            nn.UpsamplingBilinear2d(scale_factor=2),
            Conv(encoder_channels[0], num_classes, kernel_size=1)
        )

    def forward(self, inputs):
        # 假设 inputs 是 FPN 输出的特征图
        x = inputs  # 根据需要调整输入

        output = self.cls_seg(x)
        return output

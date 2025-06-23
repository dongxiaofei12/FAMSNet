# Copyright (c) OpenMMLab. All rights reserved.
from functools import partial
from itertools import chain
from typing import Sequence

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from torch.autograd import Variable
import torch.nn.functional as F
from mmcv.cnn.bricks import DropPath
from mmengine.model import BaseModule, ModuleList, Sequential

from mmseg.models.backbones.buildformer import RPE, LWMSA, Mlp, StageModule
from mmseg.models.utils import build_norm_layer, GRN, DepthwiseSeparableConvWithWTConv2d

from mmseg.models.utils.LiteMLA import LiteMLA
from mmseg.models.utils.fdlsr import WGB
from mmseg.registry import MODELS
class InceptionDWConv2d(nn.Module):
    """ Inception depthweise convolution
    """

    def __init__(self, in_channels, square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125, norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU')):
        super().__init__()

        gc = int(in_channels * branch_ratio)  # channel numbers of a convolution branch
        self.dwconv_hw = ConvModule(
            in_channels=gc,
            out_channels=gc,
            kernel_size=square_kernel_size,
            stride=1,
            dilation=1,
            padding=square_kernel_size // 2,
            groups=gc,
            conv_cfg=None,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )
        self.dwconv_w = ConvModule(
            in_channels=gc,
            out_channels=gc,
            kernel_size=(1, band_kernel_size),
            stride=1,
            dilation=1,
            padding=(0, band_kernel_size // 2),
            groups=gc,
            conv_cfg=None,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )
        self.dwconv_h = ConvModule(
            in_channels=gc,
            out_channels=gc,
            kernel_size=(band_kernel_size, 1),
            stride=1,
            dilation=1,
            padding=(band_kernel_size // 2, 0),
            groups=gc,
            conv_cfg=None,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )
        self.split_indexes = (in_channels - 3 * gc, gc, gc, gc)


    def forward(self, x):
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        x_hw = self.dwconv_hw(x_hw)
        x_w = self.dwconv_w(x_w)
        x_h = self.dwconv_h(x_h)
        out = torch.cat(
            (x_id, x_hw, x_w, x_h),
            dim=1,
        )

        return out

class WFAB(nn.Module):
    def __init__(self, in_channels=3, norm_cfg=dict(type='BN'),act_cfg=dict(type='ReLU'), window_size=16, dpr=0.1,num_heads=4):
        super().__init__()
        self.wtconv1 = DepthwiseSeparableConvWithWTConv2d(in_channels,in_channels, kernel_size=3)
        self.wtconv2 = WGB(in_channels)
        self.wtconv3 = DepthwiseSeparableConvWithWTConv2d(in_channels, in_channels, kernel_size=5)
        self.wtconv4 = WGB(in_channels)

        self.drop = DropPath(dpr)




    def forward(self, x):
        skip_x = x
        skip_x = self.wtconv1(skip_x)
        skip_x = self.wtconv2(skip_x)
        skip_x = skip_x + x
        skip_x = nn.GELU()(skip_x)

        skip_x = self.wtconv3(skip_x)
        skip_x = self.wtconv4(skip_x)
        skip_x = skip_x + x
        skip_x = nn.GELU()(skip_x)

        out = self.drop(skip_x)


        return out

class ffnblock(nn.Module):
    def __init__(self,in_channels,mlp_ratio=4.,norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU'),drop_path_rate=0.1):
        super().__init__()

        self.norm = build_norm_layer(norm_cfg, in_channels)
        mid_channels = int(mlp_ratio * in_channels)
        self.pointwise_conv1 = nn.Linear(in_channels, mid_channels)
        self.act = MODELS.build(act_cfg)
        self.pointwise_conv2 = nn.Linear(mid_channels, in_channels)
        self.drop_path = DropPath(
            drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x, data_format='channel_last')
        x = self.pointwise_conv1(x)
        x = self.act(x)
        x = self.pointwise_conv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = shortcut + self.drop_path(x)
        return x

class MSCB(nn.Module):
    def __init__(self, in_channels=3, norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU'),num_heads=4, window_size=16, dpr=0.1,reduction=16):
        super().__init__()
        # self.inception = InceptionDWConv2d(in_channels,norm_cfg=norm_cfg,act_cfg=act_cfg)
        self.conv1 = ConvModule(in_channels, in_channels, kernel_size=3,padding=1, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv2 = ConvModule(in_channels, in_channels, kernel_size=5,padding=2, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv3 = ConvModule(
            in_channels, in_channels, kernel_size=3, padding=2, dilation=2,
            norm_cfg=norm_cfg, act_cfg=act_cfg
        )
        self.conv4 = ConvModule(
            in_channels, in_channels, kernel_size=3, padding=4, dilation=4,
            norm_cfg=norm_cfg, act_cfg=act_cfg
        )
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.fuse = ConvModule(2 * in_channels,in_channels,kernel_size=1,padding=0,norm_cfg=norm_cfg,act_cfg=act_cfg)
        self.norm = build_norm_layer(norm_cfg, in_channels)
        self.mla = LiteMLA(in_channels=in_channels, out_channels=in_channels, scales=(7, 9))
        self.ffn = ffnblock(in_channels, norm_cfg=norm_cfg, act_cfg=act_cfg)

        self.fusion = DepthwiseSeparableConvModule(2 * in_channels, in_channels, kernel_size=1, norm_cfg=norm_cfg,
                                                  act_cfg=act_cfg)
        self.drop = DropPath(dpr)

    def forward(self, x):
        # out = self.inception(x)
        skip_x = x
        skip_x1 = self.conv1(skip_x)
        skip_x2 = self.conv2(skip_x)
        skip_x3 = self.conv3(skip_x)
        skip_x4 = self.conv4(skip_x)
        fus_x1 = skip_x1 + skip_x2
        fus_x2 = skip_x3 + skip_x4
        fus_x1 = self.maxpool1(fus_x1)
        fus_x2 = self.maxpool2(fus_x2)
        fus = torch.cat([fus_x1, fus_x2], dim=1)
        fus_x = self.fuse(fus)


        norm_x = self.norm(skip_x)
        mla = self.mla(norm_x)
        mla = mla + norm_x
        mla = self.ffn(mla)


        out =torch.cat([mla, fus_x], dim=1)
        out = self.fusion(out)
        out = out + x
        out = self.drop(out)

        return out


class ConvNeXtBlock(BaseModule):
    """ConvNeXt Block.

    Args:
        in_channels (int): The number of input channels.
        dw_conv_cfg (dict): Config of depthwise convolution.
            Defaults to ``dict(kernel_size=7, padding=3)``.
        norm_cfg (dict): The config dict for norm layers.
            Defaults to ``dict(type='LN2d', eps=1e-6)``.
        act_cfg (dict): The config dict for activation between pointwise
            convolution. Defaults to ``dict(type='GELU')``.
        mlp_ratio (float): The expansion ratio in both pointwise convolution.
            Defaults to 4.
        linear_pw_conv (bool): Whether to use linear layer to do pointwise
            convolution. More details can be found in the note.
            Defaults to True.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.
        layer_scale_init_value (float): Init value for Layer Scale.
            Defaults to 1e-6.

    Note:
        There are two equivalent implementations:

        1. DwConv -> LayerNorm -> 1x1 Conv -> GELU -> 1x1 Conv;
           all outputs are in (N, C, H, W).
        2. DwConv -> LayerNorm -> Permute to (N, H, W, C) -> Linear -> GELU
           -> Linear; Permute back

        As default, we use the second to align with the official repository.
        And it may be slightly faster.
    """

    def __init__(self,
                 in_channels,
                 dw_conv_cfg=dict(kernel_size=7, padding=3),
                 norm_cfg=dict(type='LN2d', eps=1e-6),
                 act_cfg=dict(type='GELU'),
                 mlp_ratio=4.,
                 linear_pw_conv=True,
                 drop_path_rate=0.,
                 layer_scale_init_value=1e-6,
                 use_grn=False,
                 with_cp=False):
        super().__init__()
        self.with_cp = with_cp

        self.depthwise_conv = nn.Conv2d(
            in_channels, in_channels, groups=in_channels, **dw_conv_cfg)

        self.linear_pw_conv = linear_pw_conv
        self.norm = build_norm_layer(norm_cfg, in_channels)

        mid_channels = int(mlp_ratio * in_channels)
        if self.linear_pw_conv:
            # Use linear layer to do pointwise conv.
            pw_conv = nn.Linear
        else:
            pw_conv = partial(nn.Conv2d, kernel_size=1)

        self.pointwise_conv1 = pw_conv(in_channels, mid_channels)
        self.act = MODELS.build(act_cfg)
        self.pointwise_conv2 = pw_conv(mid_channels, in_channels)


        if use_grn:
            self.grn = GRN(mid_channels)
        else:
            self.grn = None

        self.gamma = nn.Parameter(
            layer_scale_init_value * torch.ones((in_channels)),
            requires_grad=True) if layer_scale_init_value > 0 else None

        self.drop_path = DropPath(
            drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x):

        def _inner_forward(x):
            shortcut = x
            x = self.depthwise_conv(x)

            if self.linear_pw_conv:
                x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
                x = self.norm(x, data_format='channel_last')
                x = self.pointwise_conv1(x)
                x = self.act(x)
                if self.grn is not None:
                    x = self.grn(x, data_format='channel_last')
                x = self.pointwise_conv2(x)
                x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
            else:
                x = self.norm(x, data_format='channel_first')
                x = self.pointwise_conv1(x)
                x = self.act(x)

                if self.grn is not None:
                    x = self.grn(x, data_format='channel_first')
                x = self.pointwise_conv2(x)


            if self.gamma is not None:
                x = x.mul(self.gamma.view(1, -1, 1, 1))

            x = shortcut + self.drop_path(x)
            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x


@MODELS.register_module()
class FAMSNet(BaseModule):
    arch_settings = {
        'atto': {
            'depths': [2, 2, 6, 2],
            'channels': [40, 80, 160, 320]
        },
        'femto': {
            'depths': [2, 2, 6, 2],
            'channels': [48, 96, 192, 384]
        },
        'pico': {
            'depths': [2, 2, 6, 2],
            'channels': [64, 128, 256, 512]
        },
        'nano': {
            'depths': [2, 2, 8, 2],
            'channels': [80, 160, 320, 640]
        },
        'tiny': {
            'depths': [3, 3, 9, 3],
            'channels': [96, 192, 384, 768]
        },
        'small': {
            'depths': [3, 3, 27, 3],
            'channels': [96, 192, 384, 768]
        },
        'base': {
            'depths': [3, 3, 27, 3],
            'channels': [128, 256, 512, 1024]
        },
        'large': {
            'depths': [3, 3, 27, 3],
            'channels': [192, 384, 768, 1536]
        },
        'xlarge': {
            'depths': [3, 3, 27, 3],
            'channels': [256, 512, 1024, 2048]
        },
        'huge': {
            'depths': [3, 3, 27, 3],
            'channels': [352, 704, 1408, 2816]
        }
    }

    def __init__(self,
                 arch='tiny',
                 in_channels=3,
                 stem_patch_size=4,
                 norm_cfg=dict(type='LN2d', eps=1e-6),
                 act_cfg=dict(type='GELU'),
                 linear_pw_conv=True,
                 use_grn=False,
                 drop_path_rate=0.,
                 layer_scale_init_value=1e-6,
                 out_indices=-1,
                 frozen_stages=0,
                 gap_before_final_norm=True,
                 with_cp=False,
                 init_cfg=[
                     dict(
                         type='TruncNormal',
                         layer=['Conv2d', 'Linear'],
                         std=.02,
                         bias=0.),
                     dict(
                         type='Constant', layer=['LayerNorm'], val=1.,
                         bias=0.),
                 ]):
        super().__init__(init_cfg=init_cfg)

        if isinstance(arch, str):
            assert arch in self.arch_settings, \
                f'Unavailable arch, please choose from ' \
                f'({set(self.arch_settings)}) or pass a dict.'
            arch = self.arch_settings[arch]
        elif isinstance(arch, dict):
            assert 'depths' in arch and 'channels' in arch, \
                f'The arch dict must have "depths" and "channels", ' \
                f'but got {list(arch.keys())}.'

        self.depths = arch['depths']
        self.channels = arch['channels']
        assert (isinstance(self.depths, Sequence)
                and isinstance(self.channels, Sequence)
                and len(self.depths) == len(self.channels)), \
            f'The "depths" ({self.depths}) and "channels" ({self.channels}) ' \
            'should be both sequence with the same length.'

        self.num_stages = len(self.depths)

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = 4 + index
                assert out_indices[i] >= 0, f'Invalid out_indices {index}'
        self.out_indices = out_indices

        self.frozen_stages = frozen_stages
        self.gap_before_final_norm = gap_before_final_norm

        # stochastic depth decay rule
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(self.depths))
        ]
        block_idx = 0

        # 4 downsample layers between stages, including the stem layer.
        self.downsample_layers = ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                self.channels[0],
                kernel_size=stem_patch_size,
                stride=stem_patch_size),
            build_norm_layer(norm_cfg, self.channels[0]),
        )
        self.downsample_layers.append(stem)

        # 4 feature resolution stages, each consisting of multiple residual
        # blocks
        self.stages = nn.ModuleList()
        self.wtin = nn.ModuleList()
        self.att = nn.ModuleList()
        for i in range(self.num_stages):
            depth = self.depths[i]
            channels = self.channels[i]

            if i >= 1:
                downsample_layer = nn.Sequential(
                    build_norm_layer(norm_cfg, self.channels[i - 1]),
                    nn.Conv2d(
                        self.channels[i - 1],
                        channels,
                        kernel_size=2,
                        stride=2),
                )
                self.downsample_layers.append(downsample_layer)

            stage = Sequential(*[
                ConvNeXtBlock(
                    in_channels=channels,
                    drop_path_rate=dpr[block_idx + j],
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    linear_pw_conv=linear_pw_conv,
                    layer_scale_init_value=layer_scale_init_value,
                    use_grn=use_grn,
                    with_cp=with_cp) for j in range(depth)
            ])
            block_idx += depth

            self.stages.append(stage)
            if i < 2:
                self.wtin.append(WFAB(channels,norm_cfg=norm_cfg,act_cfg=act_cfg,num_heads=2 * (i+1), dpr=0.1 * i))
            else:
                self.att.append(MSCB(channels,norm_cfg=norm_cfg,act_cfg=act_cfg,num_heads=2 ** (i+1), dpr=0.1 * i))
            if i in self.out_indices:
                norm_layer = build_norm_layer(norm_cfg, channels)
                self.add_module(f'norm{i}', norm_layer)

        self._freeze_stages()


    def forward(self, x):
        outs = []
        for i, stage in enumerate(self.stages):
            x = self.downsample_layers[i](x)
            x = stage(x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                if self.gap_before_final_norm:
                    gap = x.mean([-2, -1], keepdim=True)
                    outs.append(norm_layer(gap).flatten(1))
                else:
                    if i < 2:
                        x1 = self.wtin[i](x)
                        outs.append(x1)
                    else:
                        x2 = self.att[i-2](x)
                        outs.append(x2)

        return tuple(outs)

    def _freeze_stages(self):
        for i in range(self.frozen_stages):
            downsample_layer = self.downsample_layers[i]
            stage = self.stages[i]
            downsample_layer.eval()
            stage.eval()
            for param in chain(downsample_layer.parameters(),
                               stage.parameters()):
                param.requires_grad = False

    def train(self, mode=True):
        super(ConvNeXtwtin, self).train(mode)
        self._freeze_stages()

    def get_layer_depth(self, param_name: str, prefix: str = ''):
        """Get the layer-wise depth of a parameter.

        Args:
            param_name (str): The name of the parameter.
            prefix (str): The prefix for the parameter.
                Defaults to an empty string.

        Returns:
            Tuple[int, int]: The layer-wise depth and the num of layers.
        """
        print(f"get_layer_depth called with param_name: {param_name}")
        max_layer_id = 12 if self.depths[-2] > 9 else 6

        if not param_name.startswith(prefix):
            # For subsequent module like head
            return max_layer_id + 1, max_layer_id + 2

        param_name = param_name[len(prefix):]
        if param_name.startswith('downsample_layers'):
            stage_id = int(param_name.split('.')[1])
            if stage_id == 0:
                layer_id = 0
            elif stage_id == 1 or stage_id == 2:
                layer_id = stage_id + 1
            else:  # stage_id == 3:
                layer_id = max_layer_id

        elif param_name.startswith('stages'):
            stage_id = int(param_name.split('.')[1])
            block_id = int(param_name.split('.')[2])
            if stage_id == 0 or stage_id == 1:
                layer_id = stage_id + 1
            elif stage_id == 2:
                layer_id = 3 + block_id // 3
            else:  # stage_id == 3:
                layer_id = max_layer_id

        # final norm layer
        else:
            layer_id = max_layer_id + 1

        return layer_id, max_layer_id + 2
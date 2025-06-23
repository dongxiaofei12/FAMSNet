import torch
from torch import nn


class WALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(WALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        if channel * 4 // reduction == 0:
            m = 1
        else:
            m = channel * 4 // reduction
        self.conv_du1 = nn.Sequential(
            nn.Conv2d(channel * 4, m, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(m, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.conv_du2 = nn.Sequential(
            nn.Conv2d(channel * 4, m, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(m, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.DWT = DWT()
        self.IWT = IWT()

    def forward(self, x):
        B, C, H, W = x.size()
        Hd = H % 2
        Wd = W % 2
        PAD = nn.ZeroPad2d(padding=(0, Wd, 0, Hd))
        x_ = PAD(x)
        x_ = self.DWT(x_)
        y_avg = self.avg_pool(x_)
        y_max = self.max_pool(x_)
        y = self.conv_du1(y_avg) + self.conv_du2(y_max)
        return x * y

class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        x01 = x[:, :, 0::2, :] / 2
        x02 = x[:, :, 1::2, :] / 2
        x1 = x01[:, :, :, 0::2]
        x2 = x02[:, :, :, 0::2]
        x3 = x01[:, :, :, 1::2]
        x4 = x02[:, :, :, 1::2]
        x_LL = x1 + x2 + x3 + x4
        x_HL = -x1 - x2 + x3 + x4
        x_LH = -x1 + x2 - x3 + x4
        x_HH = x1 - x2 - x3 + x4
        return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)

class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        r = 2
        in_batch, in_channel, in_height, in_width = x.size()
        # print([in_batch, in_channel, in_height, in_width])
        out_batch, out_channel, out_height, out_width = in_batch, int(
            in_channel / (r ** 2)), r * in_height, r * in_width
        x1 = x[:, 0:out_channel, :, :] / 2
        x2 = x[:, out_channel:out_channel * 2, :, :] / 2
        x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
        x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2

        h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().cuda()

        h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
        h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
        h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
        h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4
        return h

def dwt_init(x):
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)

def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    # print([in_batch, in_channel, in_height, in_width])
    out_batch, out_channel, out_height, out_width = in_batch, int(
        in_channel / (r ** 2)), r * in_height, r * in_width
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2

    h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().cuda()

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h


class WGB(nn.Module):
    def __init__(self, channel, reduction=16):
        super(WGB, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        if channel * 4 // reduction == 0:
            m = 1
        else:
            m = channel * 4 // reduction
        self.conv_du1 = nn.Sequential(
            nn.Conv2d(channel * 4, m, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(m, channel * 4, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.conv_du2 = nn.Sequential(
            nn.Conv2d(channel * 4, m, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(m, channel * 4, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.DWT = DWT()
        self.IWT = IWT()
        self.gate = nn.Sigmoid()
        self.WA = WALayer(channel)

    def forward(self, x):
        B, C, H, W = x.size()
        Hd = H % 2
        Wd = W % 2
        PAD = nn.ZeroPad2d(padding=(0, Wd, 0, Hd))
        x = self.WA(x)
        x_ = PAD(x)
        x_ = self.DWT(x_)
        y_avg = self.avg_pool(x_)
        y_max = self.max_pool(x_)
        y = self.conv_du1(y_avg) + self.conv_du2(y_max)
        y = y * x_
        map = self.IWT(y)
        map_ = map[:, :, 0:H, 0:W]
        y = self.gate(map_)
        return x * y  # + self.WA(x)

class WGB_woWA(nn.Module):
    def __init__(self, channel, reduction=16):
        super(WGB_woWA, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        if channel * 4 // reduction == 0:
            m = 1
        else:
            m = channel * 4 // reduction
        self.conv_du1 = nn.Sequential(
            nn.Conv2d(channel * 4, m, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(m, channel * 4, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.conv_du2 = nn.Sequential(
            nn.Conv2d(channel * 4, m, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(m, channel * 4, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.DWT = DWT()
        self.IWT = IWT()
        self.gate = nn.Sigmoid()
        self.WA = WALayer(channel)

    def forward(self, x):
        B, C, H, W = x.size()
        Hd = H % 2
        Wd = W % 2
        PAD = nn.ZeroPad2d(padding=(0, Wd, 0, Hd))
        x = self.WA(x)
        x_ = PAD(x)
        x_ = self.DWT(x_)
        y_avg = self.avg_pool(x_)
        y_max = self.max_pool(x_)
        y = self.conv_du1(y_avg) + self.conv_du2(y_max)
        y = y * x_
        map = self.IWT(y)
        map_ = map[:, :, 0:H, 0:W]
        y = self.gate(map_)
        return x * y  # + self.WA(x)



    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))

if __name__ == '__main__':
    input_data = torch.randn(1, 3, 256, 256).cuda()
    model = WGB(3).cuda()
    output_data = model(input_data)
    # 打印输入和输出的形状
    print("Input shape:", input_data.shape)
    print("Output shape:", output_data.shape)
import torch
import torch.nn as nn


# Github地址：https://github.com/JIAOJIAYUASD/dilateformer
# 论文地址：https://arxiv.org/abs/2302.01791
class DilateAttention(nn.Module):
    "Implementation of Dilate-attention"

    def __init__(self, head_dim, qk_scale=None, attn_drop=0, kernel_size=3, dilation=1):
        super().__init__()
        self.head_dim = head_dim
        self.scale = qk_scale or head_dim ** -0.5
        self.kernel_size = kernel_size
        self.unfold = nn.Unfold(kernel_size, dilation=dilation, padding=dilation * (kernel_size - 1) // 2)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, q, k, v):
        B, d, H, W = q.shape

        # 计算 q
        q = q.reshape(B, -1, H * W).permute(0, 2, 1).unsqueeze(2)  # [B, H*W, 1, d]

        # 计算 k 和 v
        k = self.unfold(k).reshape(B, d, -1, H * W).permute(0, 3, 2, 1)  # [B, H*W, K*K, d]
        v = self.unfold(v).reshape(B, d, -1, H * W).permute(0, 3, 2, 1)  # [B, H*W, K*K, d]

        # 计算注意力
        attn = (q @ k.transpose(-1, -2)) * self.scale  # [B, H*W, 1, K*K]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # 应用注意力到 v
        x = (attn @ v).squeeze(2).permute(0, 2, 1).reshape(B, d, H, W)
        return x



class MultiDilatelocalAttention(nn.Module):
    "Implementation of Dilate-attention"

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., kernel_size=3, dilation=[2, 3]):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.scale = qk_scale or self.head_dim ** -0.5
        self.num_dilation = len(dilation)
        assert num_heads % self.num_dilation == 0, f"num_heads{num_heads} must be divisible by num_dilation{self.num_dilation}!!"
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=qkv_bias)
        self.dilate_attention = nn.ModuleList(
            [DilateAttention(self.head_dim, qk_scale, attn_drop, kernel_size, dilation[i])
             for i in range(self.num_dilation)])
        self.proj = nn.Conv2d(dim, dim, 1)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, C, H, W = x.shape  # 输入维度为 [B, C, H, W]

        # 计算 qkv 并调整形状
        qkv = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, H, W)
        qkv = qkv.permute(1, 0, 2, 3, 4, 5)  # [3, B, num_heads, head_dim, H, W]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 将 q, k, v 分割为不同的 dilation 组
        q = q.reshape(B, self.num_dilation, -1, self.head_dim, H, W)
        k = k.reshape(B, self.num_dilation, -1, self.head_dim, H, W)
        v = v.reshape(B, self.num_dilation, -1, self.head_dim, H, W)

        x_out = []
        for i in range(self.num_dilation):
            qi = q[:, i].reshape(-1, self.head_dim, H, W)  # [B * heads_per_dilation, head_dim, H, W]
            ki = k[:, i].reshape(-1, self.head_dim, H, W)
            vi = v[:, i].reshape(-1, self.head_dim, H, W)
            xi = self.dilate_attention[i](qi, ki, vi)
            x_out.append(xi)

        # 将结果拼接并恢复形状
        x = torch.cat(x_out, dim=0)
        x = x.reshape(B, -1, H, W)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x



if __name__ == "__main__":
    x = torch.rand([8, 512, 32, 32]).cuda() #输入B C H W
    m = MultiDilatelocalAttention(512).cuda()
    y = m(x)
    print(y.shape)
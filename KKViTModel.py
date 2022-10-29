# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 17:08:23 2022

@author: PC
"""

import torch
import torch.nn as nn
from functools import partial
from .functions import ReverseLayerF

# ---- ---- ---- ---- ---- 网络结构部分 ---- ---- ---- ---- ---- #
def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class PatchEmbed(nn.Module):
    """
    将一维振动信号转化为矩阵形式
    """
    def __init__(self,embed_dim=250):
        super(PatchEmbed, self).__init__()
        # self.T = nn.Tanh()
        self.l1 = nn.Linear(embed_dim,embed_dim)
        
    def forward(self, x):
        B, C, HW = x.shape
        x = self.l1(x)
        # x = self.T(x)
        return x

class Attention(nn.Module):
    def __init__(self,
                 dim,   # 输入token的dim
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = x.shape

        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self,
                 dim=250,
                 num_heads=10,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.1,
                 attn_drop_ratio=0.1,
                 drop_path_ratio=0.1,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class VisionTransformer(nn.Module):
    def __init__(self, 
                 num_patches=100,
                 embed_dim=100,
                 num_heads=10,
                 num_classes=4,
                 depth=1, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, drop_ratio=0.1,
                 attn_drop_ratio=0.1, drop_path_ratio=0.01, embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None):
        """
        Args:
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_ratio (float): dropout rate
            attn_drop_ratio (float): attention dropout rate
            drop_path_ratio (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
        """
        super(VisionTransformer, self).__init__()
        # 参数设置
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.patch_embed = embed_layer(embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)
        # 模块堆积
        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        
    def forward(self, x):
        # [B, C, H, W] -> [B, num_patches, embed_dim]
        x = self.patch_embed(x)  # [B, 40, 250]
        # [1, 1, 250] -> [B, 1, 250]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)  # [B, 41, 250]
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        # x = self.head(x[:, 0])
        return x

class Classifier(nn.Module):
    def __init__(self, embed_dim = 250, num_classes = 4):
        super(Classifier, self).__init__()
        # 输出矩阵维度是batchsize * num_patches * embed_dim
        # 全连接神经网络需要维度需要和最后一个参数对齐
        self.head = nn.Sequential(
                nn.Linear(embed_dim,320),
                nn.Linear(320,160),
                nn.Linear(160,num_classes)
                )
        
    def forward(self,x):
        
        # Droupout+分类
        x = self.head(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, embed_dim = 250):
        super(Discriminator, self).__init__()
        
        self.layer = nn.Sequential(
                nn.Linear(embed_dim,320),
                nn.Linear(320,160),
                nn.Linear(160,1),
                nn.Sigmoid()
                )
        
    def forward(self,x):
        
        x = self.layer(x)
        return x

# class DAN(nn.Module):
#     def __init__(self, num_patches=50, embed_dim=100, num_heads=10, num_classes=4):
#         super(DAN, self).__init__()
#         self.sharednet = VisionTransformer(num_patches=50, embed_dim=100, num_heads=10, num_classes=4)
#         self.classnet = Classifier(embed_dim=100, num_classes=4)
#         self.domainnet = Discriminator(embed_dim=100)

#     def forward(self, data, alpha=1):
        
#         data = self.sharednet(data)
#         reverse_data = ReverseLayerF.apply(data[:, 0], alpha)
#         class_out = self.classnet(data[:, 0])
#         domain_out = self.domainnet(reverse_data)
        
#         return class_out, domain_out


class DAN(nn.Module):
    def __init__(self, num_patches=100, embed_dim=100, num_heads=10, num_classes=4):
        super(DAN, self).__init__()
        self.sharednet = VisionTransformer(num_patches=num_patches, embed_dim=embed_dim, num_heads=num_heads, num_classes=num_classes)
        self.classnet = Classifier(embed_dim=embed_dim, num_classes=num_classes)
        self.domainnet = Discriminator(embed_dim=embed_dim)

    def forward(self, data, alpha=1):
        
        data = self.sharednet(data)
        reverse_data = ReverseLayerF.apply(data[:, 0], alpha)
        class_out = self.classnet(data[:, 0])
        domain_out = self.domainnet(reverse_data)
        
        return class_out, domain_out

# class DAN(nn.Module):
#     def __init__(self, num_patches=100, embed_dim=100, num_heads=10, num_classes=4):
#         super(DAN, self).__init__()
#         self.sharednet = VisionTransformer(num_patches=100, embed_dim=100, num_heads=10, num_classes=4)
#         self.classnet = Classifier(embed_dim=100, num_classes=4)
#         self.domainnet = Discriminator(embed_dim=100)

#     def forward(self, data, alpha=1):
        
#         data = self.sharednet(data)
#         reverse_data = ReverseLayerF.apply(data, alpha)
#         class_out = self.classnet(data[:, 0])
#         domain_out = self.domainnet(reverse_data[:, 0])
        
#         return class_out, domain_out

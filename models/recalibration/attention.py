import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_channels, eps=1e-5, momentum=0.01, affine=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        # global-pooling along the HxW-dim
        x_compress = self.avg_pool(x)
        # calc attention
        x_attention = self.attention(x_compress)
        # scale/refine
        return x * x_attention


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size):
        super(SpatialAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, stride=1, padding=int(kernel_size/2), bias=False),
            nn.BatchNorm2d(1, eps=1e-5, momentum=0.01, affine=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        # global-pooling along the C-dim
        x_compress = torch.cat((torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1)
        # calc attention
        x_attention = self.attention(x_compress)
        # scale/refine
        return x * x_attention


class HMAttention(nn.Module):
    '''Hierarchical Mixed Attention'''
    def __init__(self, in_channels, att_types, kernel_size):
        super(HMAttention, self).__init__()
        self.att_types = att_types
        # create attention module
        for att_type in self.att_types:
            if att_type=='channel':
                self.channel_attention = ChannelAttention(in_channels)
            elif att_type=='spatial':
                self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        # attention
        for att_type in self.att_types:
            if att_type=='channel':
                x = self.channel_attention(x)
            elif att_type=='spatial':
                x = self.spatial_attention(x)
        # return
        return x
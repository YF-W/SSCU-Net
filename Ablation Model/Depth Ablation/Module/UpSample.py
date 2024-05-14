import torch
import torch.nn as nn

from .Transition import Transition


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels, size=4):
        super().__init__()

        self.interpolate_bilinear = nn.Upsample(scale_factor=size, mode='bilinear', align_corners=True)
        self.conv_transpose = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=size, stride=size)
        self.transition = Transition(in_channels * 2, out_channels)
        self.bn = nn.BatchNorm2d(out_channels)
        self.elu = nn.ELU(inplace=True)

    def forward(self, x):
        upsample_1 = self.interpolate_bilinear(x)
        upsample_2 = self.conv_transpose(x)
        x = torch.cat([upsample_1, upsample_2], axis=1)
        x = self.transition(x)
        x = self.bn(x)
        x = self.elu(x)

        return x

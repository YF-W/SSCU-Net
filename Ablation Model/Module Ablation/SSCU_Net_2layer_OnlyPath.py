import torch
import torch.nn as nn

from Module.DoubleConv import DoubleConv
from Module.DirectionalConvPath import DirectionalConvPath


class NET(nn.Module):
    def __init__(self, in_channels, out_channels, channels=32):
        super().__init__()

        self.down_1 = DoubleConv(in_channels, channels)
        self.down_2 = DoubleConv(channels, channels * 4)
        self.downsample = nn.MaxPool2d(kernel_size=4, stride=4)
        self.directionalconvpath_1 = DirectionalConvPath(channels)
        self.directionalconvpath_2 = DirectionalConvPath(channels * 4)
        self.bottleneck = DoubleConv(channels * 4, channels * 8)
        self.upsample_1 = nn.ConvTranspose2d(channels * 8, channels * 4, kernel_size=4, stride=4)
        self.upsample_2 = nn.ConvTranspose2d(channels * 4, channels, kernel_size=4, stride=4)
        self.up_1 = DoubleConv(channels * 8, channels * 4)
        self.up_2 = DoubleConv(channels * 2, channels)
        self.final_conv = nn.Conv2d(channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.down_1(x)
        connection_1 = self.directionalconvpath_1(x)
        x = self.downsample(x)
        x = self.down_2(x)
        connection_2 = self.directionalconvpath_2(x)
        x = self.downsample(x)
        x = self.bottleneck(x)
        x = self.upsample_1(x)
        x = torch.cat([x, connection_2], axis=1)
        x = self.up_1(x)
        x = self.upsample_2(x)
        x = torch.cat([x, connection_1], axis=1)
        x = self.up_2(x)
        x = self.final_conv(x)

        return x

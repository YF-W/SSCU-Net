import torch
import torch.nn as nn

from Module.SuperConvBlock import SuperConvBlock
from Module.DownSample import DownSample
from Module.DirectionalConvPath import DirectionalConvPath
from Module.UpSample import UpSample


class SSCUNET(nn.Module):
    def __init__(self, in_channels, out_channels, channels=64):
        super().__init__()

        self.superconvblock_down = SuperConvBlock(in_channels, channels)
        self.downsample = DownSample(channels)
        self.directionalconvpath = DirectionalConvPath(channels)
        self.bottleneck = SuperConvBlock(channels, channels * 4)
        self.upsample = UpSample(channels * 4, channels)
        self.superconvblock_up = SuperConvBlock(channels * 2, channels)
        self.final_conv = nn.Conv2d(channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.superconvblock_down(x)
        connection = self.directionalconvpath(x)
        x = self.downsample(x)
        x = self.bottleneck(x)
        x = self.upsample(x)
        x = torch.cat([x, connection], axis=1)
        x = self.superconvblock_up(x)
        x = self.final_conv(x)

        return x

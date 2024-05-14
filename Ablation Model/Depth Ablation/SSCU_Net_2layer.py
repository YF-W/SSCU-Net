import torch
import torch.nn as nn

from Module.SuperConvBlock import SuperConvBlock
from Module.DownSample import DownSample
from Module.DirectionalConvPath import DirectionalConvPath
from Module.UpSample import UpSample


class SSCUNET(nn.Module):
    def __init__(self, in_channels, out_channels, channels=32):
        super().__init__()

        self.superconvblock_down_1 = SuperConvBlock(in_channels, channels)
        self.superconvblock_down_2 = SuperConvBlock(channels, channels * 4)
        self.downsample_1 = DownSample(channels)
        self.downsample_2 = DownSample(channels * 4)
        self.restore = nn.ConvTranspose2d(channels * 4, channels, kernel_size=4, stride=4)
        self.directionalconvpath_1 = DirectionalConvPath(channels)
        self.directionalconvpath_2 = DirectionalConvPath(channels * 4)
        self.bottleneck = SuperConvBlock(channels * 4, channels * 8)
        self.upsample_1 = UpSample(channels * 8, channels * 4)
        self.upsample_2 = UpSample(channels * 4, channels)
        self.superconvblock_up_1 = SuperConvBlock(channels * 8, channels * 4)
        self.superconvblock_up_2 = SuperConvBlock(channels * 2, channels)
        self.final_conv = nn.Conv2d(channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.superconvblock_down_1(x)
        connection_1 = x
        x = self.downsample_1(x)
        x = self.superconvblock_down_2(x)
        restore = self.restore(x)
        connection_1 = self.directionalconvpath_1(connection_1 + restore)
        connection_2 = self.directionalconvpath_2(x)
        x = self.downsample_2(x)
        x = self.bottleneck(x)
        x = self.upsample_1(x)
        restore = self.restore(x)
        connection_1 = connection_1 + restore
        x = torch.cat([x, connection_2], axis=1)
        x = self.superconvblock_up_1(x)
        x = self.upsample_2(x)
        x = torch.cat([x, connection_1], axis=1)
        x = self.superconvblock_up_2(x)
        x = self.final_conv(x)

        return x

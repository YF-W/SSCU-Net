import torch
import torch.nn as nn

from Module.SuperConvBlock import SuperConvBlock


class NET(nn.Module):
    def __init__(self, in_channels, out_channels, channels=32):
        super().__init__()

        self.superconvblock_down_1 = SuperConvBlock(in_channels, channels)
        self.superconvblock_down_2 = SuperConvBlock(channels, channels * 4)
        self.downsample = nn.MaxPool2d(kernel_size=4, stride=4)
        self.bottleneck = SuperConvBlock(channels * 4, channels * 8)
        self.upsample_1 = nn.ConvTranspose2d(channels * 8, channels * 4, kernel_size=4, stride=4)
        self.upsample_2 = nn.ConvTranspose2d(channels * 4, channels, kernel_size=4, stride=4)
        self.superconvblock_up_1 = SuperConvBlock(channels * 8, channels * 4)
        self.superconvblock_up_2 = SuperConvBlock(channels * 2, channels)
        self.final_conv = nn.Conv2d(channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.superconvblock_down_1(x)
        connection_1 = x
        x = self.downsample(x)
        x = self.superconvblock_down_2(x)
        connection_2 = x
        x = self.downsample(x)
        x = self.bottleneck(x)
        x = self.upsample_1(x)
        x = torch.cat([x, connection_2], axis=1)
        x = self.superconvblock_up_1(x)
        x = self.upsample_2(x)
        x = torch.cat([x, connection_1], axis=1)
        x = self.superconvblock_up_2(x)
        x = self.final_conv(x)

        return x

import torch
import torch.nn as nn

from Module.DoubleConv import DoubleConv


class NET(nn.Module):
    def __init__(self, in_channels, out_channels, channels=32):
        super().__init__()
        self.down_1 = DoubleConv(in_channels, channels)
        self.down_2 = DoubleConv(channels, channels * 4)
        self.restore = nn.ConvTranspose2d(channels * 4, channels, kernel_size=4, stride=4)
        self.bottleneck = DoubleConv(channels * 4, channels * 8)
        self.upsample_1 = nn.ConvTranspose2d(channels * 8, channels * 4, kernel_size=4, stride=4)
        self.up_1 = DoubleConv(channels * 8, channels * 4)
        self.upsample_2 = nn.ConvTranspose2d(channels * 4, channels, kernel_size=4, stride=4)
        self.up_2 = DoubleConv(channels * 2, channels)
        self.pool = nn.MaxPool2d(kernel_size=4, stride=4)
        self.final_conv = nn.Conv2d(channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.down_1(x)
        connection_1 = x
        x = self.pool(x)
        x = self.down_2(x)
        restore = self.restore(x)
        connection_1 = connection_1 + restore
        connection_2 = x
        x = self.pool(x)
        x = self.bottleneck(x)
        x = self.upsample_1(x)
        restore = self.restore(x)
        connection_1 = connection_1 + restore
        concat_skip_1 = torch.cat([x, connection_2], dim=1)
        x = self.up_1(concat_skip_1)
        x = self.upsample_2(x)
        concat_skip_2 = torch.cat([x, connection_1], dim=1)
        x = self.up_2(concat_skip_2)

        return self.final_conv(x)

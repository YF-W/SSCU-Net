import torch
import torch.nn as nn
from thop import profile


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.transition = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ELU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        )

    def forward(self, x):
        return self.transition(x)


class DownSample(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.max_pool = nn.MaxPool2d(kernel_size=4, stride=4)
        self.avg_pool = nn.AvgPool2d(kernel_size=4, stride=4)
        self.transition = Transition(channels * 2, channels)
        self.bn = nn.BatchNorm2d(channels)
        self.elu = nn.ELU(inplace=True)

    def forward(self, x):
        dowmsample_1 = self.max_pool(x)
        dowmsample_2 = self.avg_pool(x)
        x = torch.cat([dowmsample_1, dowmsample_2], axis=1)
        x = self.transition(x)
        x = self.bn(x)
        x = self.elu(x)

        return x


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.interpolate_bilinear = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.conv_transpose = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=4, stride=4)
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


class NET(nn.Module):
    def __init__(self, in_channels, out_channels, channels=32):
        super().__init__()

        self.down_1 = DoubleConv(in_channels, channels)
        self.down_2 = DoubleConv(channels, channels * 4)
        self.downsample_1 = DownSample(channels)
        self.downsample_2 = DownSample(channels * 4)
        self.bottleneck = DoubleConv(channels * 4, channels * 8)
        self.upsample_1 = UpSample(channels * 8, channels * 4)
        self.upsample_2 = UpSample(channels * 4, channels)
        self.up_1 = DoubleConv(channels * 8, channels * 4)
        self.up_2 = DoubleConv(channels * 2, channels)
        self.final_conv = nn.Conv2d(channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.down_1(x)
        connection_1 = x
        x = self.downsample_1(x)
        x = self.down_2(x)
        connection_2 = x
        x = self.downsample_2(x)
        x = self.bottleneck(x)
        x = self.upsample_1(x)
        x = torch.cat([x, connection_2], axis=1)
        x = self.up_1(x)
        x = self.upsample_2(x)
        x = torch.cat([x, connection_1], axis=1)
        x = self.up_2(x)
        x = self.final_conv(x)

        return x

import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding='same', dilation=1, function='normal',
                 activation='T'):
        super().__init__()

        self.function = function
        self.activation = activation
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding)
        self.atrous_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, dilation=dilation)
        self.depth_conv = nn.Conv2d(in_channels, in_channels, kernel_size * 3, stride, padding=padding,
                                    groups=in_channels)
        self.point_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, groups=1)
        self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        if self.function == 'normal':
            x = self.conv(x)
        elif self.function == 'atrous':
            x = self.atrous_conv(x)
        elif self.function == 'depthwise':
            x = self.depth_conv(x)
            x = self.point_conv(x)

        x = self.batch_norm(x)

        if self.activation == 'T':
            return F.elu(x)
        else:
            return x


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


class SuperConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv_1x1 = ConvBlock(in_channels, out_channels, kernel_size=1, stride=1, padding='same', dilation=1,
                                  function='normal', activation='T')
        self.conv_3x3 = ConvBlock(in_channels, out_channels, kernel_size=3, stride=1, padding='same', dilation=1,
                                  function='normal', activation='T')
        self.conv_5x5 = ConvBlock(in_channels, out_channels, kernel_size=5, stride=1, padding='same', dilation=1,
                                  function='normal', activation='T')
        self.conv_7x7 = ConvBlock(in_channels, out_channels, kernel_size=7, stride=1, padding='same', dilation=1,
                                  function='normal', activation='T')
        self.transition = Transition(out_channels * 3, out_channels)
        self.atrous_conv_3x3_2 = ConvBlock(in_channels, out_channels, kernel_size=3, stride=1, padding='same',
                                           dilation=2, function='atrous', activation='T')
        self.atrous_conv_3x3_4 = ConvBlock(out_channels, out_channels, kernel_size=3, stride=1, padding='same',
                                           dilation=4, function='atrous', activation='T')
        self.atrous_conv_3x3_6 = ConvBlock(out_channels, out_channels, kernel_size=3, stride=1, padding='same',
                                           dilation=6, function='atrous', activation='T')
        self.depthwise_conv_3x3_before = ConvBlock(in_channels, out_channels, kernel_size=3, stride=1, padding='same',
                                                   dilation=1, function='depthwise', activation='T')
        self.depthwise_conv_3x3_after = ConvBlock(out_channels, out_channels, kernel_size=3, stride=1, padding='same',
                                                  dilation=1, function='depthwise', activation='T')
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv_transpose = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)
        self.final_transition = Transition(out_channels * 5, out_channels)
        self.bn = nn.BatchNorm2d(out_channels)
        self.elu = nn.ELU(inplace=True)

    def forward(self, x):
        layer_1 = self.conv_1x1(x)

        layer_2_1 = self.conv_3x3(x)
        layer_2_2 = self.conv_5x5(x)
        layer_2_3 = self.conv_7x7(x)
        layer_2 = torch.cat([layer_2_1, layer_2_2, layer_2_3], axis=1)
        layer_2 = self.transition(layer_2)

        layer_3 = self.atrous_conv_3x3_2(x)
        layer_3 = self.atrous_conv_3x3_4(layer_3)
        layer_3 = self.atrous_conv_3x3_6(layer_3)

        layer_4 = self.depthwise_conv_3x3_before(x)
        layer_4 = self.depthwise_conv_3x3_after(layer_4)
        layer_4 = self.depthwise_conv_3x3_after(layer_4)

        layer_5 = self.avg_pool(x)
        layer_5 = self.conv_1x1(layer_5)
        layer_5 = self.conv_transpose(layer_5)

        x = torch.cat([layer_1, layer_2, layer_3, layer_4, layer_5], axis=1)
        x = self.final_transition(x)
        x = self.bn(x)
        x = self.elu(x)

        return x


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

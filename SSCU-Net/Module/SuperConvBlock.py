import torch
import torch.nn as nn

from .ConvBlock import ConvBlock
from .Transition import Transition


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

import torch.nn as nn

from .ConvBlock import ConvBlock
from .DirectionalConvBlock import DirectionalConvBlock


class DirectionalConvPath(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.shortcut = ConvBlock(channels, channels, kernel_size=1, stride=1, padding='same', dilation=1,
                                  function='normal', activation='F')
        self.directional_conv = DirectionalConvBlock(channels)
        self.bn = nn.BatchNorm2d(channels)
        self.elu = nn.ELU(inplace=True)

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.directional_conv(x)
        x = x + shortcut
        x = self.bn(x)
        x = self.elu(x)

        return x

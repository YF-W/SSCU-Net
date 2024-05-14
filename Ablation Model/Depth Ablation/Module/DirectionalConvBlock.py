import torch.nn as nn

from .DirectionalConvLayer import DirectionalConvLayer


class DirectionalConvBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.conv_vertical = DirectionalConvLayer(channels=channels, kernel_size=(1, 9), direction='vertical')
        self.conv_horizontal = DirectionalConvLayer(channels=channels, kernel_size=(9, 1), direction='horizontal')
        self.bn = nn.BatchNorm2d(channels)
        self.elu = nn.ELU(inplace=True)

    def forward(self, x):
        x_vertical = self.conv_vertical(x)
        x_horizontal = self.conv_horizontal(x)
        x = x_vertical + x_horizontal
        x = self.bn(x)
        x = self.elu(x)

        return x

import torch.nn as nn
import torch.nn.functional as F


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

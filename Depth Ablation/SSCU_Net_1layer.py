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


class DirectionalConvLayer(nn.Module):
    def __init__(self, channels, kernel_size, direction='vertical'):
        super().__init__()

        self.direction = direction

        self.conv_vertical = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, stride=1,
                      padding=((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2)),
            nn.InstanceNorm2d(channels),
            nn.ELU(inplace=True)
        )

        self.conv_horizontal = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, stride=1,
                      padding=((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2)),
            nn.InstanceNorm2d(channels),
            nn.ELU(inplace=True)
        )

    def forward(self, x):
        n, c, h, w = x.size()
        feature_stack = []

        if self.direction == 'vertical':
            for i in range(h):
                feature_i = x.select(2, i).reshape(n, c, 1, w)
                if i == 0:
                    feature_stack.append(feature_i)
                    continue
                feature_stack.append(self.conv_vertical(feature_stack[i - 1]) + feature_i)

            for i in range(h):
                pos = h - i - 1
                if pos == h - 1:
                    continue
                feature_stack[pos] = self.conv_vertical(feature_stack[pos + 1]) + feature_stack[pos]
                self.conv_vertical(feature_stack[pos + 1])

            x = torch.cat(feature_stack, 2)

        elif self.direction == 'horizontal':
            for i in range(w):
                feature_i = x.select(3, i).reshape(n, c, h, 1)
                if i == 0:
                    feature_stack.append(feature_i)
                    continue
                feature_stack.append(self.conv_horizontal(feature_stack[i - 1]) + feature_i)

            for i in range(w):
                pos = w - i - 1
                if pos == w - 1:
                    continue
                feature_stack[pos] = self.conv_horizontal(feature_stack[pos + 1]) + feature_stack[pos]

            x = torch.cat(feature_stack, 3)

        return x


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

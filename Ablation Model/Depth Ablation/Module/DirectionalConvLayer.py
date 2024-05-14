import torch
import torch.nn as nn


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

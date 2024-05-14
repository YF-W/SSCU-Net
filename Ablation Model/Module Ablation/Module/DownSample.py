import torch
import torch.nn as nn

from .Transition import Transition


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

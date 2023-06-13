"""resnet in pytorch
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.
    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385v1
"""

import torch
import torch.nn as nn
from net.BAM import BAM


class CsCp(nn.Module):

    def __init__(self, in_channels):
        super().__init__()

        self.in_channels = in_channels

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True))

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        # self.BAM = BAM(in_channels)
        self.BAM1 = BAM(in_channels)
        self.BAM2 = BAM(in_channels)
        self.BAM3 = BAM(in_channels)
        self.compress = nn.Conv2d(in_channels*2, in_channels, (3, 3), stride=(1, 1), padding=(1, 1))

        self.sigmoid = nn.Sigmoid()

        # self.conv_ = nn.Conv2d(512*2*2*2, 256*2*2*2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x1, x2):

        x = torch.cat((x1, x2), 1)
        x = self.compress(x)

        x, att = self.BAM3(x)
        # print(x.shape)
        x1_r = x1 * (1 - self.sigmoid(att))
        x1_r, att_1 = self.BAM1(x1_r)
        x2_r = x2 * (1 - self.sigmoid(att))
        x2_r, att_2 = self.BAM2(x2_r)

        out = x1_r + x2_r + x

        return out


if __name__ == '__main__':
    m = CsCp(32)
    input1 = torch.randn(4, 32, 50, 50)
    input2 = torch.randn(4, 32, 50, 50)
    output = m(input1, input2)
    print(output.shape)

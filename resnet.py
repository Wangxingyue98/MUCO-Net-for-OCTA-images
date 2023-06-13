"""resnet in pytorch
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.
    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385v1
"""

import torch
import torch.nn as nn
from net.BAM import BAM
from net.CsCp import CsCp
from net.crossViewFusion import CrossViewFusion

class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34
    """

    #BasicBlock and BottleNeck block
    #have different output size
    #we use class attribute expansion
    #to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers
    """
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class ResNet(nn.Module):

    def __init__(self, block, num_block, num_classes=3):
        super().__init__()

        self.in_channels1 = 64
        self.in_channels2 = 64
        block1 = block
        num_block1 = num_block
        block2 = block
        num_block2 = num_block

        self.conv11 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True))
        self.conv12 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True))
        #we use a different inputsize than the original paper
        #so conv2_x's stride is 1
        self.conv21_x = self._make_layer(block1, 64, num_block1[0], 1)
        self.conv22_x = self.__make_layer(block2, 64, num_block2[0], 1)

        self.conv31_x = self._make_layer(block1, 128, num_block1[1], 2)
        self.conv32_x = self.__make_layer(block2, 128, num_block2[1], 2)

        self.conv41_x = self._make_layer(block1, 256, num_block1[2], 2)
        self.conv42_x = self.__make_layer(block2, 256, num_block2[2], 2)

        self.conv51_x = self._make_layer(block1, 512, num_block1[3], 2)
        self.conv52_x = self.__make_layer(block2, 512, num_block2[3], 2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.fc_1 = nn.Linear(2048, num_classes)
        self.BAM = BAM(2048)
        self.fc_2 = nn.Linear(2048, num_classes)
        self.fc_3 = nn.Linear(2048, num_classes)

        # self.CsCp_1 = CsCp(64)
        self.CsCp_1 = CsCp(64)
        self.CsCp_2 = CsCp(256)
        self.CsCp_3 = CsCp(512)
        self.CsCp_4 = CsCp(1024)
        self.CsCp_5 = CsCp(2048)

        self.sigmoid = nn.Sigmoid()

        self.compress11 = nn.Conv2d(64 * 2, 64, kernel_size=3, padding=1, bias=False)
        self.compress12 = nn.Conv2d(64 * 2, 64, kernel_size=3, padding=1, bias=False)

        self.compress21 = nn.Conv2d(256 * 2, 256, kernel_size=3, padding=1, bias=False)
        self.compress22 = nn.Conv2d(256 * 2, 256, kernel_size=3, padding=1, bias=False)

        self.compress31 = nn.Conv2d(512 * 2, 512, kernel_size=3, padding=1, bias=False)
        self.compress32 = nn.Conv2d(512 * 2, 512, kernel_size=3, padding=1, bias=False)

        self.compress41 = nn.Conv2d(1024 * 2, 1024, kernel_size=3, padding=1, bias=False)
        self.compress42 = nn.Conv2d(1024 * 2, 1024, kernel_size=3, padding=1, bias=False)

        self.compress5 = nn.Conv2d(2048 * 3, 2048, kernel_size=3, padding=1, bias=False)
        
        self.cross1 = CrossViewFusion([64, 192, 192], [256, 192, 192])
        self.cross2 = CrossViewFusion([256, 192, 192], [512, 96, 96])
        self.cross3 = CrossViewFusion([512, 96, 96], [1024, 48, 48])
        self.cross4 = CrossViewFusion([1024, 48, 48], [2048, 24, 24])

        # self.conv_ = nn.Conv2d(512*2*2*2, 256*2*2*2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def _make_layer(self, block1, out_channels, num_blocks1, stride1):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block
        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer
        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides1 = [stride1] + [1] * (num_blocks1 - 1)
        layers1 = []
        for stride1 in strides1:
            layers1.append(block1(self.in_channels1, out_channels, stride1))
            self.in_channels1 = out_channels * block1.expansion

        return nn.Sequential(*layers1)

    def __make_layer(self, block2, out_channels, num_blocks2, stride2):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block
        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer
        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides2 = [stride2] + [1] * (num_blocks2 - 1)
        layers2 = []
        for stride2 in strides2:
            layers2.append(block2(self.in_channels2, out_channels, stride2))
            self.in_channels2 = out_channels * block2.expansion

        return nn.Sequential(*layers2)

    def forward(self, x1, x2):

        x1 = self.conv11(x1)
        x2 = self.conv12(x2)
        out_1 = self.CsCp_1(x1, x2)  # 64, 192, 192
        # print(x1.shape, x2.shape)

        x1 = self.compress11(torch.cat((x1, out_1), 1))
        x2 = self.compress12(torch.cat((x2, out_1), 1))
        x1 = self.conv21_x(x1)
        x2 = self.conv22_x(x2)
        out_2 = self.CsCp_2(x1, x2)  # 256, 192, 192
        out_2 = self.cross1(out_1, out_2)

        x1 = self.compress21(torch.cat((x1, out_2), 1))
        x2 = self.compress22(torch.cat((x2, out_2), 1))
        x1 = self.conv31_x(x1)
        x2 = self.conv32_x(x2)
        out_3 = self.CsCp_3(x1, x2)  # 512, 96, 96
        out_3 = self.cross2(out_2, out_3)

        x1 = self.compress31(torch.cat((x1, out_3), 1))
        x2 = self.compress32(torch.cat((x2, out_3), 1))
        x1 = self.conv41_x(x1)
        x2 = self.conv42_x(x2)
        out_4 = self.CsCp_4(x1, x2)  # 1024, 48, 48
        out_4 = self.cross3(out_3, out_4)

        x1 = self.compress41(torch.cat((x1, out_4), 1))
        x2 = self.compress42(torch.cat((x2, out_4), 1))
        x1 = self.conv51_x(x1)
        x2 = self.conv52_x(x2)
        out_5 = self.CsCp_5(x1, x2)  # 2048, 24, 24
        out_5 = self.cross4(out_4, out_5)

        output_5, _ = self.BAM(out_5)
        output_5 = self.avg_pool(output_5)
        output_5 = output_5.view(output_5.size(0), -1)
        output_5 = self.fc_1(output_5)

        # 直接连接
        output = self.compress5(torch.cat((x1, x2, out_5), 1))
        # output = out_5
        # print(out_1.shape, out_2.shape, out_3.shape, out_4.shape, out_5.shape, output.shape)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
 
        a_1 = self.fc_2(self.avg_pool(x1).view(x1.size(0), -1))
        a_2 = self.fc_3(self.avg_pool(x2).view(x2.size(0), -1))

        return output, output_5, a_1, a_2

def resnet18():
    """ return a ResNet 18 object
    """
    return ResNet(BasicBlock, [2, 2, 2, 2])

def resnet34():
    """ return a ResNet 34 object
    """
    return ResNet(BasicBlock, [3, 4, 6, 3])

def resnet50():
    """ return a ResNet 50 object
    """
    return ResNet(BottleNeck, [3, 4, 6, 3])

def resnet101():
    """ return a ResNet 101 object
    """
    return ResNet(BottleNeck, [3, 4, 23, 3])

def resnet152():
    """ return a ResNet 152 object
    """
    return ResNet(BottleNeck, [3, 8, 36, 3])

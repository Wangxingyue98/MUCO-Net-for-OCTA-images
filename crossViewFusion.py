import torch
import torch.nn as nn
from net.BAM import BAM
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce


class CrossViewFusion(nn.Module):

    def __init__(self, x1shape, x2shape):
        super().__init__()

        # self.in_channels = in_channels

        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(3, in_channels, kernel_size=3, padding=1, bias=False),
        #     nn.BatchNorm2d(in_channels),
        #     nn.ReLU(inplace=True))
        self.shape_1 = x1shape
        self.shape_2 = x2shape
        C_1, H_1, W_1 = self.shape_1
        C_2, H_2, W_2 = self.shape_2
        self.softmax = nn.Softmax(dim=-1)
        # self.fc_1 = nn.Linear(C_1*H_2*W_2, C_1*H_2*W_2)
        # self.fc_2 = nn.Linear(C*H*W, C*H*W)

        self.q = nn.Conv2d(C_2, C_2, kernel_size=1, bias=False)
        self.k = nn.Conv2d(C_1, C_1, kernel_size=1, bias=False)
        self.v = nn.Conv2d(C_1, C_1, kernel_size=1, bias=False)

    def forward(self, x1, x2):

        C, H, W = self.shape_2

        _x2 = x2

        x1 = reduce(x1, 'b c (h1 h2) (w1 w2) -> b c h1 w1', 'mean', h1=H, w1=W)

        x1T  = self.k(x1)
        x1T = rearrange(x1T, 'b e (h) (w) -> b e (h w)')
        x1T = x1T.transpose(1, 2)

        x2 = self.q(x2)
        x2 = rearrange(x2, 'b e (h) (w) -> b e (h w)')

        out = torch.bmm(x2, x1T, out=None)
        out = out / H
        out = self.softmax(out)
        out = torch.bmm(out, rearrange(self.v(x1), 'b e (h) (w) -> b e (h w)'))
        out = out.reshape(-1, C, H, W)
        out = _x2 + out

        return out


if __name__ == '__main__':
    # m = nn.AdaptiveAvgPool2d((1, 1))
    # input1 = torch.randn(4, 32, 50, 50)
    # input2 = torch.randn(4, 32, 50, 50)
    # output = m(input1)
    # print(output.shape)
    #

    x1 = torch.randn(4, 24, 92, 92)
    x2 = torch.randn(4, 48, 92//2, 92//2)
    B_1, C_1, H_1, W_1 = x1.size()
    B_2, C_2, H_2, W_2 = x2.size()
    m = CrossViewFusion([C_1, H_1, W_1], [C_2, H_2, W_2])
    out = m(x1, x2)
    print(out.shape)

    # x11 = reduce(x1, 'b c (h1 h2) (w1 w2) -> b c h1 w1', 'mean', h1=H, w1=W)
    # x11 = rearrange(x11, 'b e (h) (w) -> b e (h w)')
    # x22 = rearrange(x2, 'b e (h) (w) -> b e (h w)')
    # print(x1.shape, x11.shape)
    # print(x2.shape, x22.shape)
    # x1T = x11.transpose(1, 2)
    # print(x1T.shape)
    # out = torch.bmm(x22, x1T, out=None)
    # print(out.shape)
    # out = torch.bmm(out, x11)
    # print(out.shape)
    # out = out.reshape(-1, C, H, W)
    # print(out.shape)
    # out = out / H
    # print(out.shape)
    # m = nn.Softmax()
    # out = m(out)
    # print(out.shape)

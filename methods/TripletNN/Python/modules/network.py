import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

import math 
import numpy as np

class SharedAtrousConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__()
        self.weights = nn.Parameter(torch.rand(int(out_channels/2), in_channels, 3, 3))
        nn.init.kaiming_uniform_(self.weights, mode='fan_out', nonlinearity='relu')
        if bias:
            self.bias1 = nn.Parameter(torch.zeros(int(out_channels/2)))
            self.bias2 = nn.Parameter(torch.zeros(int(out_channels/2)))
        else:
            self.bias1 = None
            self.bias2 = None
    def forward(self, x):
        x1 = nn.functional.conv2d(x, self.weights, stride=1, padding='same', bias=self.bias1)
        x2 = nn.functional.conv2d(x, self.weights, stride=1, padding='same', dilation=2, bias=self.bias2)
        x3 = torch.cat([x1, x2], 1)
        return x3

class SharedAtrousResBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.conv_res = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.net = nn.Sequential(
            SharedAtrousConv2d(in_channels, middle_channels, bias=False),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            SharedAtrousConv2d(middle_channels, out_channels, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        res = self.conv_res(x)
        x = self.net(x)
        x = (x + res) * (1 / math.sqrt(2))
        x = self.relu(x)
        return x
    
class Resize(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None, antialias=False):
        super().__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        self.recompute_scale_factor = recompute_scale_factor
        self.antialias = antialias
    def forward(self, x):
        return torch.nn.functional.interpolate(x, size=self.size, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners, recompute_scale_factor=self.recompute_scale_factor, antialias=self.antialias)
        
class NestedSharedAtrousResUNet(nn.Module):
    def __init__(self, num_classes, num_channels, width=32, resolution=(240, 320)):
        super().__init__()
        self.resolution = resolution
        nb_filter = [width, width*2, width*4, width*8, width*16]

        self.pool = Resize(scale_factor=0.5, mode='bilinear')
        self.up = Resize(scale_factor=2, mode='bilinear')

        self.conv0_0 = SharedAtrousResBlock(num_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = SharedAtrousResBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = SharedAtrousResBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = SharedAtrousResBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = SharedAtrousResBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = SharedAtrousResBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = SharedAtrousResBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = SharedAtrousResBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = SharedAtrousResBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = SharedAtrousResBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = SharedAtrousResBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = SharedAtrousResBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = SharedAtrousResBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = SharedAtrousResBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = SharedAtrousResBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0]*4, num_classes, kernel_size=1)

    def forward(self, input):
        
        x0_0 = self.conv0_0(input) #320x240
        x1_0 = self.conv1_0(self.pool(x0_0)) #160x120
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0)) #80x60
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0)) #40x30
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0)) #20x15
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))
        
        output = self.final(torch.cat([x0_1, x0_2, x0_3, x0_4], 1))
        
        return output

class fclayer(nn.Module):
    def __init__(self, in_h = 8, in_w = 10, out_n = 6):
        super().__init__()
        self.in_h = in_h
        self.in_w = in_w
        self.out_n = out_n
        self.fc_list = []
        for i in range(out_n):
            self.fc_list.append(nn.Linear(in_h*in_w, 6))
        self.fc_list = nn.ModuleList(self.fc_list)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(36, 6)
    def forward(self, x):
        x = x.reshape(-1, self.out_n, self.in_h, self.in_w)
        outs = []
        for i in range(self.out_n):
            outs.append(self.fc_list[i](x[:, i, :, :].reshape(-1, self.in_h*self.in_w)))
        out = torch.cat(outs, 1)
        out = self.act(out)
        out = self.fc2(out)
        return out

class conv(nn.Module):
    def __init__(self, in_channels=512, out_n = 6):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_n, kernel_size=1, stride=1, padding='same')
    def forward(self, x):
        x = self.conv(x)
        return x
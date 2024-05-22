""" Vision Transformer (ViT) in PyTorch
Hacked together by / Copyright 2020 Ross Wightman
Adapted from https://github.com/hila-chefer/Transformer-Explainability
"""
import torch
import torch.nn as nn
import numpy as np
import math
from einops import rearrange
from modules.layers_ours import *

from helpers import load_pretrained
from weight_init import trunc_normal_
from layer_helpers import to_2tuple

import torch.nn as nn

# setup a resnet block and its forward function
class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, kernel_size=3, doSkip=True, doBn=True):
        super(ResNetBlock, self).__init__()
        self.conv1 = Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1, bias=False)
        self.bn1 = BatchNorm2d(out_channels)
        self.acti1 = ReLU()
        self.conv2 = Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=1, bias=False)
        self.bn2 = BatchNorm2d(out_channels)
        self.acti2 = ReLU()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        
        self.doSkip = doSkip
        self.doBn = doBn

        self.add = Add()   

        self.conv3 = Conv2d(in_channels, out_channels, kernel_size=1, stride=stride,bias=False)
        self.bn3 =  BatchNorm2d(out_channels)

        self.clone = Clone()
        
    def forward(self, x):
        if self.doSkip:
            ai1, ai2 = self.clone(x, 2)
        else:
            ai1 = x
        out = self.conv1(ai1)
        if self.doBn:
            out = self.bn1(out)
        out = self.acti1(out)
        self.conv2(out)
        if self.doBn:
            out = self.bn2(out)
        if self.doSkip:
            if self.stride != 1 or self.in_channels != self.out_channels:
                out2 = self.conv3(ai2)
                if self.doBn:
                    out2 = self.bn3(out2)
            else:
                out2 = ai2
            out = self.add([out, out2])

        out = self.acti2(out)
        return out

    def relprop(self, cam, **kwargs):
        if self.doSkip:
            (cam1, cam2) = self.add.relprop(cam, **kwargs)
            if self.doBn:
                cam2 = self.bn3.relprop(cam2, **kwargs)
            cam2 = self.conv3.relprop(cam2, **kwargs)
        else: 
            cam1 = cam
        if self.doBn:
            cam1 = self.bn2.relprop(cam1, **kwargs)
        cam1 = self.conv2.relprop(cam1, **kwargs)
        if self.doBn:
            cam1 = self.bn1.relprop(cam1, **kwargs)
        cam1 = self.conv1.relprop(cam1, **kwargs)

        if self.doSkip:
            cam = self.clone.relprop((cam1, cam2), **kwargs)
        else:
            cam = cam1

        return cam

# setup the final model structure
class ResNetLikeClassifier(nn.Module):
    
    def __init__(self, num_hidden_layers=2, nc=1, nf=8, outputs=2, dropout=0.0, maskValue = -2, stride=1, kernel_size=3, inDim=6, doSkip=True, doBn=True):
        super(ResNetLikeClassifier, self).__init__()
        self.dmodel = nf
        self.maskValue = maskValue
        self.finalDim = nf*(pow(2, num_hidden_layers-1))

        self.resnet_blocks = nn.ModuleList([ResNetBlock(nc, nf, stride=stride, kernel_size=kernel_size, doSkip=doSkip, doBn=doBn) if i == 0 else ResNetBlock(nf*(pow(2, i-1)), nf*(pow(2, i) ), stride=stride, kernel_size=kernel_size)  for i in range(num_hidden_layers)])

        self.lastConv = Conv2d(self.finalDim, self.finalDim, kernel_size=1, stride=1, bias=False)
        self.lastFF = Linear((inDim) * self.finalDim, (inDim) * self.finalDim)
        self.flatten = nn.Flatten()
        self.out = Linear((inDim) * self.finalDim, outputs)

        self.doDropout = dropout > 0
        self.dropout = Dropout(dropout)
        self.actiOut = SIGMOID()

    def forward(self, input):
        output = input
        for i, layer_module in enumerate(self.resnet_blocks):
            output = layer_module(output)

        if self.doDropout:
            output = self.flatten(output)
            output = self.lastFF(output)
            output = self.dropout(output)
        else:
            output = self.lastConv(output)
            output = self.flatten(output)

        output = self.out(output)
        output = self.actiOut(output)

        return output

    def relprop(self, cam, method=None, **kwargs):
        cam = self.out.relprop(cam, **kwargs)
        if self.doDropout:
            cam = self.lastFF.relprop(cam, **kwargs)
            cam = cam.reshape((cam.shape[0], self.finalDim ,-1 , 1)) #unflatten
        else:
            cam = cam.reshape((cam.shape[0], self.finalDim ,-1 , 1)) #unflatten
            cam = self.lastConv.relprop(cam, **kwargs)

        for layer_module in reversed(self.resnet_blocks):
            cam = layer_module.relprop(cam, **kwargs)
        return cam

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.models as models
import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from unet_parts import *
from skimage.filters import threshold_otsu
from PIL import Image

class UNetDet(nn.Module):
    """
    CNN module for clinically significant prostate cancer detection. 
    Used a the same architecture as U-Net by replacing the decoder part of the network by a classification layer instead. 

    """

    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNetDet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)
        self.down5 = Down(512, 512)
        self.flat1 =  nn.Conv2d(512, 128, kernel_size = 3, stride=1, padding=0)
        self.classifier = nn.Conv2d(128, n_classes, kernel_size = 1, stride=1, padding=0)

    def forward(self, x):

        x1 = self.inc(x)
        # 96
        
        x2 = self.down1(x1)
        # 48
        
        x3 = self.down2(x2)
        # 24
        
        
        x4 = self.down3(x3)
        # 12 
        
        x5 = self.down4(x4)
        # 6 
        
        x6 = self.down5(x5)
        # 3 
        
        x7 = self.flat1(x6)
        # 1 
        
        x8 = self.classifier(x7)
        # 1 
        
        return x8

    
  

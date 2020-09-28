
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

class Modified3DUNet(nn.Module):
    def __init__(self, in_channels, n_classes, base_n_filter = 8):
        super(Modified3DUNet, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.base_n_filter = base_n_filter

        self.lrelu = nn.LeakyReLU()
        self.dropout3d = nn.Dropout3d(p=0.6)
        self.upsacle = nn.Upsample(scale_factor=2, mode='nearest')
#         self.upsacle = nn.functional.interpolate(scale_factor=2, mode='nearest')
        self.softmax = nn.Softmax(dim=1)

        # Level 1 context pathway
        self.conv3d_c1_1 = nn.Conv3d(self.in_channels, self.base_n_filter, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3d_c1_2 = nn.Conv3d(self.base_n_filter, self.base_n_filter, kernel_size=3, stride=1, padding=1, bias=False)
        self.lrelu_conv_c1 = self.lrelu_conv(self.base_n_filter, self.base_n_filter)
        self.inorm3d_c1 = nn.InstanceNorm3d(self.base_n_filter)
        
    
        # Level 2 context pathway
        self.conv3d_c2 = nn.Conv3d(self.base_n_filter, self.base_n_filter*2, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm_lrelu_conv_c2 = self.norm_lrelu_conv(self.base_n_filter*2, self.base_n_filter*2)
        self.inorm3d_c2 = nn.InstanceNorm3d(self.base_n_filter*2)

        # Level 3 context pathway
        self.conv3d_c3 = nn.Conv3d(self.base_n_filter*2, self.base_n_filter*4, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm_lrelu_conv_c3 = self.norm_lrelu_conv(self.base_n_filter*4, self.base_n_filter*4)
        self.inorm3d_c3 = nn.InstanceNorm3d(self.base_n_filter*4)

    
        # Level 4 context pathway
        self.conv3d_c4 = nn.Conv3d(self.base_n_filter*4, self.base_n_filter*8, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm_lrelu_conv_c4 = self.norm_lrelu_conv(self.base_n_filter*8, self.base_n_filter*8)
        self.inorm3d_c4 = nn.InstanceNorm3d(self.base_n_filter*8)

        # Level 5 context pathway, level 0 localization pathway
        self.conv3d_c5 = nn.Conv3d(self.base_n_filter*8, self.base_n_filter*16, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm_lrelu_conv_c5 = self.norm_lrelu_conv(self.base_n_filter*16, self.base_n_filter*16)
        self.norm_lrelu_upscale_conv_norm_lrelu_l0 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter*16, self.base_n_filter*8)

        self.conv3d_l0 = nn.Conv3d(self.base_n_filter*8, self.base_n_filter*8, kernel_size = 1, stride=1, padding=0, bias=False)
        self.inorm3d_l0 = nn.InstanceNorm3d(self.base_n_filter*8)

        # Level 1 localization pathway
        self.conv_norm_lrelu_l1 = self.conv_norm_lrelu(self.base_n_filter*16, self.base_n_filter*16)
        self.conv3d_l1 = nn.Conv3d(self.base_n_filter*16, self.base_n_filter*8, kernel_size=1, stride=1, padding=0, bias=False)
        self.norm_lrelu_upscale_conv_norm_lrelu_l1 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter*8, self.base_n_filter*4)

        # Level 2 localization pathway
        self.conv_norm_lrelu_l2 = self.conv_norm_lrelu(self.base_n_filter*8, self.base_n_filter*8)
        self.conv3d_l2 = nn.Conv3d(self.base_n_filter*8, self.base_n_filter*4, kernel_size=1, stride=1, padding=0, bias=False)
        self.norm_lrelu_upscale_conv_norm_lrelu_l2 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter*4, self.base_n_filter*2)

        # Level 3 localization pathway
        self.conv_norm_lrelu_l3 = self.conv_norm_lrelu(self.base_n_filter*4, self.base_n_filter*4)
        self.conv3d_l3 = nn.Conv3d(self.base_n_filter*4, self.base_n_filter*2, kernel_size=1, stride=1, padding=0, bias=False)
        self.norm_lrelu_upscale_conv_norm_lrelu_l3 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter*2, self.base_n_filter)

        # Level 4 localization pathway
        self.conv_norm_lrelu_l4 = self.conv_norm_lrelu(self.base_n_filter*2, self.base_n_filter*2)
        self.conv3d_l4 = nn.Conv3d(self.base_n_filter*2, self.n_classes, kernel_size=1, stride=1, padding=0, bias=False)

        self.ds2_1x1_conv3d = nn.Conv3d(self.base_n_filter*8, self.n_classes, kernel_size=1, stride=1, padding=0, bias=False)
        self.ds3_1x1_conv3d = nn.Conv3d(self.base_n_filter*4, self.n_classes, kernel_size=1, stride=1, padding=0, bias=False)

    def conv_norm_lrelu(self, feat_in, feat_out):
        return nn.Sequential(
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(feat_out),
            nn.LeakyReLU())

    def norm_lrelu_conv(self, feat_in, feat_out):
        return nn.Sequential(
            nn.InstanceNorm3d(feat_in),
            nn.LeakyReLU(),
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False))

    def lrelu_conv(self, feat_in, feat_out):
        return nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False))

    def norm_lrelu_upscale_conv_norm_lrelu(self, feat_in, feat_out):
        return nn.Sequential(
            nn.InstanceNorm3d(feat_in),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
#             nn.functional.interpolate(scale_factor=2, mode='nearest'),

            # should be feat_in*2 or feat_in
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(feat_out),
            nn.LeakyReLU())

    def forward(self, x):
        
        #  Level 1 context pathway
        out = self.conv3d_c1_1(x)
        residual_1 = out
        out = self.lrelu(out)
        out = self.conv3d_c1_2(out)
        out = self.dropout3d(out)
        out = self.lrelu_conv_c1(out)
        # Element Wise Summation
        out += residual_1
        context_1 = self.lrelu(out)
        out = self.inorm3d_c1(out)
        out = self.lrelu(out)

        # Level 2 context pathway
        out = self.conv3d_c2(out)
        residual_2 = out
        out = self.norm_lrelu_conv_c2(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c2(out)
        out += residual_2
        out = self.inorm3d_c2(out)
        out = self.lrelu(out)
        context_2 = out

        # Level 3 context pathway
        out = self.conv3d_c3(out)
        residual_3 = out
        out = self.norm_lrelu_conv_c3(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c3(out)
        out += residual_3
        out = self.inorm3d_c3(out)
        out = self.lrelu(out)
        context_3 = out

        
        # Level 4 context pathway
        out = self.conv3d_c4(out)
        residual_4 = out
        out = self.norm_lrelu_conv_c4(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c4(out)
        out += residual_4
        out = self.inorm3d_c4(out)
        out = self.lrelu(out)
        context_4 = out

        # Level 5
        out = self.conv3d_c5(out)
        residual_5 = out
        out = self.norm_lrelu_conv_c5(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c5(out)
        out += residual_5
        
    
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l0(out)

        out = self.conv3d_l0(out)
        out = self.inorm3d_l0(out)
        out = self.lrelu(out)

        # Level 1 localization pathway
        out = torch.cat([out, context_4], dim=1)
        out = self.conv_norm_lrelu_l1(out)
        out = self.conv3d_l1(out)
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l1(out)

        # Level 2 localization pathway
        out = torch.cat([out, context_3], dim=1)
        out = self.conv_norm_lrelu_l2(out)
        ds2 = out
        out = self.conv3d_l2(out)
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l2(out)

        # Level 3 localization pathway
        out = torch.cat([out, context_2], dim=1)
        out = self.conv_norm_lrelu_l3(out)
        ds3 = out
        out = self.conv3d_l3(out)
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l3(out)

        # Level 4 localization pathway
        out = torch.cat([out, context_1], dim=1)
        out = self.conv_norm_lrelu_l4(out)
        out_pred = self.conv3d_l4(out)
        
        ds2_1x1_conv = self.ds2_1x1_conv3d(ds2)
        ds1_ds2_sum_upscale = self.upsacle(ds2_1x1_conv)
        ds3_1x1_conv = self.ds3_1x1_conv3d(ds3)
        ds1_ds2_sum_upscale_ds3_sum = ds1_ds2_sum_upscale + ds3_1x1_conv
        ds1_ds2_sum_upscale_ds3_sum_upscale = self.upsacle(ds1_ds2_sum_upscale_ds3_sum)

        out = out_pred + ds1_ds2_sum_upscale_ds3_sum_upscale
        seg_layer = out
        out = out.permute(0, 2, 3, 4, 1).contiguous().view(-1, self.n_classes)
        #out = out.view(-1, self.n_classes)
#         out = self.softmax(out)
        out = nn.functional.softmax(out,dim=1)
        return out, seg_layer
    


class UNetDet(nn.Module):
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

    
    
    
class UNetDetMinusOne(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNetDetMinusOne, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)
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
        
        x6 = nn.MaxPool2d(2)(x5)
        # 3
        
        x7 = self.flat1(x6)
        # 1 
        
        x8 = self.classifier(x7)
        # 1 
        
        return x8

    
    
class UNetDetMinusTwo(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNetDetMinusTwo, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
#         self.down4 = Down(256, 512)
#         self.down5 = Down(512, 512)
        self.flat1 =  nn.Conv2d(256, 128, kernel_size = 3, stride=1, padding=0)
        self.classifier = nn.Conv2d(128, n_classes, kernel_size = 1, stride=1, padding=0)

    def forward(self, x):

        x1 = self.inc(x)
        # 96
        
        x2 = self.down1(x1)
        # 48
        
        x3 = self.down2(x2)
        # 24
        
        
        x4 = nn.MaxPool2d(2)(x3)
        # 12
        
        x5 = self.down3(x4)
        # 6 
        
        x6 = nn.MaxPool2d(2)(x5)
        # 3
        
        
        x7 = self.flat1(x6)
        # 1 
        
        x8 = self.classifier(x7)
        # 1 
        
        return x8

    
class UNetDetPlusOne(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNetDetPlusOne, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)
        self.down5 = Down(512, 512)
        self.down6 = DownNOMP(512, 512)
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
        
        x7 =  self.down6(x6)
        # 3
        
        
        x8 = self.flat1(x7)
        # 1 
        
        x9 = self.classifier(x8)
        # 1 
        
        return x9

    
    
class UNetDetPlusTwo(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNetDetPlusTwo, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)
        self.down5 = Down(512, 512)
        self.down6 = DownNOMP(512, 512)
        self.down7 = DownNOMP(512, 512)
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
        
        x7 =  self.down6(x6)
        # 3
        
        x8 =  self.down7(x7)
        # 3
        
        
        x9 = self.flat1(x8)
        # 1 
        
        x10 = self.classifier(x9)
        # 1 
        
        return x10
    
    
    
class UNetDetWOBN(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNetDetWOBN, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 32)
        self.down1 = DownWOBN(32, 64)
        self.down2 = DownWOBN(64, 128)
        self.down3 = DownWOBN(128, 256)
        self.down4 = DownWOBN(256, 512)
        self.down5 = DownWOBN(512, 512)
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
    
    
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

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
        
        x = self.up1(x5, x4)
        # 12
        
        x = self.up2(x, x3)
        # 24
        
        x = self.up3(x, x2)
        # 48
        
        x = self.up4(x, x1)
        # 96
        
        logits = self.outc(x)
        
        return logits
    

def DiceLoss(input, target):
    smooth = 1.

    iflat = input.reshape(-1)

    tflat = target.reshape(-1)
    intersection = (iflat * tflat).sum()
    
    return 1 - ((2. * intersection + smooth) /
              (iflat.sum() + tflat.sum() + smooth))


class ProstateDatasetHDF5(Dataset):
    def __init__(self, fname,transforms = None):
        self.fname=fname
        self.tables = h5py.File(fname,'r', libver='latest', swmr=True)
        self.nitems=self.tables['data'].shape[0]
        self.tables.close()
        self.data = None
        self.mask = None
        self.names = None
        self.transforms = transforms
         
    def __getitem__(self, index):
                
        self.tables = h5py.File(self.fname,'r', libver='latest', swmr=True)
        self.data = self.tables['data']
        self.mask = self.tables['mask']
        
        if "names" in self.tables:
            self.names = self.tables['names']

        img = self.data[index,1,:,:]
        mask = self.mask[index,:,:]
                
        if self.names is not None:
            name = self.names[index]
            
        self.tables.close()

        
        if self.transforms is not None:
            img = Image.fromarray(img)
            
            plt.imshow(img)
            plt.show()
            
            img = img.convert('L')
            img = img.convert('RGB')
            img = self.transforms(img)

        
        if mask.sum() == 0:
            label = 0 
        else:
            label = 1
        
        
        return img,(label,name)

    def __len__(self):
        return self.nitems
    


import h5py
import torchvision.models as models
from torch import nn
import torch 
from gradcam import GradCAM, GradCAMpp
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt 
import cv2
from skimage import measure 
import os 
from joblib import Parallel, delayed
from skimage.morphology import binary_erosion, selem
from skimage.transform import resize
from unet_parts import *
from torchvision import transforms
from gradcam import GradCAM, GradCAMpp
import torch.nn.functional as F


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
        x2 = self.down1(x1)        
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)        
        h = x5.register_hook(self.activations_hook)
        x6 = self.down5(x5)        
        x7 = self.flat1(x6)        
        x8 = self.classifier(x7)
        return x8

    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad

    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients
    
    # method for the activation exctraction
    def get_activations(self, x):

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        return x5
    
    def load_model(self,path):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint)


def get_model(path):

    model  = UNetDet(1,2)
    model.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))
    model.eval()
    return model 


def overlay_heatmap(orgimg,heatmap,mask):

    """
    orgimg : Image on which the heatmap has to be overlaid 
    heatmap : heatmap image 
    mask : The area of the image in which the heatmap has to be displayed
    """

    plt.figure(figsize=(10,10))
    plt.imshow(orgimg, cmap = 'gray',vmin =0, vmax = 1)
    plt.imshow(heatmap, cmap = 'jet', interpolation=None, vmin = 0, vmax = 1, alpha = 0.5)

    plt.xticks([])
    plt.yticks([])

    return plt 

def overlay_mask(orgimg,mask):

    """
    Mask overlaid as a contour on the input image
    orgimg : Image on which the heatmap has to be overlaid 
    mask : mask image for which the contour has to be overlaid 
    """

    contours = measure.find_contours(mask, 0)

    plt.figure(figsize=(10,10))
    plt.imshow(orgimg, cmap = 'gray', vmin = 0, vmax = 1)

    for n, contour in enumerate(contours):
        plt.plot(contour[:, 1], contour[:, 0], linewidth=3, color='r')

    plt.xticks([])
    plt.yticks([])

    return plt 


def get_unet_heat_map(model,timg,lb):

    """
    GRAD CAM based activation maps
    model : model with pretrained weights. 
    timg : input image to the model 
    lb : class for which the activation map has to be generated. 
    """


    pred = model(timg)
    pred = F.softmax(pred,dim = 1)

    # generate backward gradients 
    pred[:, lb].backward()

    gradients = model.get_activations_gradient()

    pooled_gradients =  torch.mean(gradients, dim = 0)
    pooled_gradients =  torch.mean(pooled_gradients, dim = 1)
    pooled_gradients =  torch.mean(pooled_gradients, dim = 1)

    activations = model.get_activations(timg).detach()

    for i in range(activations.shape[1]):
        activations[:, i, :, :] *= pooled_gradients[i]


    heatmap = torch.mean(activations, dim=1).squeeze()
    heatmap = np.maximum(heatmap, 0)

    # normalize the heatmap
    heatmap /= torch.max(heatmap)

    heatmap = heatmap.detach().data.cpu().numpy()
    heatmap = resize(heatmap, (96,96))

    return heatmap 


if __name__ == "__main__":

    # patch size 
    psize = 96

    # test and retest scans 
    scans = [1,2]
     
    # name of the dataset 
    dataset = "cspca"

    # number of cross validation set 
    cv = 0 


    # output foldername 
    outputfolder = fr"activationmaps"

    if not os.path.exists(outputfolder):
        os.makedirs(outputfolder)

    # looping through test and re-test scans 
    for scan in scans:

        f1path = fr"outputs\hdf5\{dataset}_{scan}_1\test.h5"
        f1 = h5py.File(f1path,libver='latest')

        d1 = f1["data"]
        mask1 = f1["mask"]
        orgmask1 = f1["orgmask"]
        names1 = f1["names"]

    
        m1path  = fr"modelcheckpoints\{dataset}\unet_{scan}_1\checkpoint.pt"
        
        m1 = get_model(m1path)
        
        activationmaps = None 


        for sample in range(len(d1)):

            name = str(names1[sample],'utf-8')
            print(name)

            orgimg = d1[sample,:,:]
            orgmask = orgmask1[sample,:,:]

            orgimg_ = np.copy(orgimg)

            img = orgimg_[None][None]
            img = torch.from_numpy(img)
            img = img.float()

            mask = mask1[sample,:,:]

            mask = mask.astype(np.float32)

            # label = 1 as we are generate heatmaps for clinically significant prostate cancer
            heatmap = get_unet_heat_map(m1,img,1)
            heatmap = np.ma.masked_where(heatmap < 0.2, heatmap)

            orgimg = np.array(Image.fromarray(orgimg).resize((psize,psize),Image.BILINEAR))
            mask = np.array(Image.fromarray(mask).resize((psize,psize),Image.NEAREST))

            orgimg = orgimg.astype(np.float32)

            mask = mask.astype(np.float32)


            singlemap =  np.vstack((orgimg[None],mask[None],heatmap[None]))[None]

            # save the input image, and the activation maps .
            if scan == 1:
                plt = overlay_mask(orgimg_,mask)
                plt.savefig(fr"{outputfolder}\{name}_org.png", bbox_inches = 'tight',pad_inches = 0)
                plt.close()

            plt = overlay_heatmap(orgimg_,heatmap,mask)

            plt.savefig(fr"{outputfolder}\{name}_{scan}.png", bbox_inches = 'tight',pad_inches = 0)
            plt.close()



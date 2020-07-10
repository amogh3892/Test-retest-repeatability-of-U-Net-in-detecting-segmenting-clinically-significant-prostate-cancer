import SimpleITK as sitk 
from pathlib import Path
import sys 
sys.path.append(fr"..\Code_general")
from dataUtil import DataUtil 
import tables
import os 
import numpy as np
from glob import glob 
import matplotlib.pyplot as plt 
from augmentation3DUtil import Augmentation3DUtil
from augmentation3DUtil import Transforms
from skimage.measure import regionprops
from PIL import Image
import pandas as pd 
from segUtil import UNet,ProstateDatasetHDF5
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.models as models
from torch.autograd import Variable
import SimpleITK as sitk 
from skimage.transform import resize as resizeImage
from skimage.filters import threshold_otsu

def _getAugmentedData(img,masks,nosamples):
    
    """ 
    This function defines different augmentations/transofrmation sepcified for a single image 
    img : to be provided SimpleITK images 
    masks : list of binary masks that has to be transformed along with the input img
    nosamples : (int) number of augmented samples to be returned
    
    """
    au = Augmentation3DUtil(img,masks=masks)

    au.add(Transforms.SHEAR,probability = 0.25, magnitude = (0.05,0.05))
    au.add(Transforms.SHEAR,probability = 0.25, magnitude = (0.02,0.05))
    au.add(Transforms.SHEAR,probability = 0.25, magnitude = (0.01,0.05))
    au.add(Transforms.SHEAR,probability = 0.25, magnitude = (0.03,0.05))

    au.add(Transforms.TRANSLATE,probability = 0.25, offset = (2,2,0))
    au.add(Transforms.TRANSLATE,probability = 0.25, offset = (-2,-2,0))
    au.add(Transforms.TRANSLATE,probability = 0.25, offset = (5,5,0))
    au.add(Transforms.TRANSLATE,probability = 0.25, offset = (-5,-5,0))

    au.add(Transforms.ROTATE2D,probability = 0.2, degrees = 3)
    au.add(Transforms.ROTATE2D,probability = 0.2, degrees = -3)
    au.add(Transforms.ROTATE2D,probability = 0.2, degrees = 5)
    au.add(Transforms.ROTATE2D,probability = 0.2, degrees = -5)
    au.add(Transforms.ROTATE2D,probability = 0.2, degrees = 8)
    au.add(Transforms.ROTATE2D,probability = 0.2, degrees = -8)

    au.add(Transforms.FLIPHORIZONTAL,probability = 0.75)

    img, augs = au.process(nosamples)

    return img,augs

def getAugmentedData(folderpath,modality, nosamples = None):
    
    
    """
    folderpath : path to folder containing images, mask
    modality : T2W/ ADC
    nosamples : The number of augmented samples to be generated
    """
    folderpath = Path(folderpath)

    try:
        ext = folderpath.glob(fr"{modality}*").__next__().stem.split(".")[-1]
    except:
        import pdb 
        pdb.set_trace()

    if ext == "gz":
        ext = ".".join(glob(fr"{folderpath}\**")[0].split("\\")[-1].split(".")[-2:])

    img = sitk.ReadImage(str(folderpath.joinpath(fr"{modality}.{ext}")))

    pm = sitk.ReadImage(str(folderpath.joinpath(fr"PM.{ext}")))
    pm = DataUtil.convert2binary(pm)

    ls = sitk.ReadImage(str(folderpath.joinpath(fr"LS.{ext}")))

    ret = []
    
    orgimg,augs = _getAugmentedData(img,[pm,ls],nosamples)
    ret.append((orgimg))

    if augs is not None:
        for i in range(len(augs)):
            ret.append(augs[i])

    return ret

def normalizeImage(img,_min,_max):

    """
    Normalize the image between (0,1) given the min and max values 
    img: The image to be normalized
    _min, _max : min and max values to be used for normalization
    """

    imgarr = sitk.GetArrayFromImage(img)
    imgarr[imgarr < _min] = _min
    imgarr[imgarr > _max] = _max

    imgarr = (imgarr - _min)/(_max - _min)
    imgarr = imgarr

    imgarr = imgarr.astype(np.float32)

    return imgarr


def getPatches(img,pm,ls,_min,_max,patchSize):

    """ 
    Collect samples from the cropped volume and add them into HDF5 file 
    img : SimpleITK image to be pre-processed
    pm : prostate mask as SimpleITK image to be preprocesed
    ls : Lesion segmentation volume, a binary mask 
    _min : Min value for the image to be normalized to
    _max : Max value for the image to be normalized to 
    patchSize: size of the patches to be resampled to. 


    returns:
    ipatches: Image patches
    opatches: Patches of organ mask 
    mpatches: patches of lesion mask
    cords: co-ordinates of the extracted patch.
    """

    cnt = 0 
    imgarr = normalizeImage(img,_min,_max)
    pmarr = sitk.GetArrayFromImage(pm)
    lsarr = sitk.GetArrayFromImage(ls)

    size = imgarr.shape

    ipatches = mpatches = opatches = None  

    cords = [] 


    for i in range(1,size[0]-1):

        mask = lsarr[i]
        orgmask = pmarr[i]

        if orgmask.sum() != 0 :

            props = regionprops(orgmask)
            (startx,starty,endx,endy) = props[0].bbox

            mpatch = mask[startx:endx, starty:endy]
            ipatch = imgarr[i,startx:endx, starty:endy]
            opatch = orgmask[startx:endx, starty:endy]

            ipatch = resizeImage(ipatch,(patchSize,patchSize))[None]

            opatch = resizeImage(opatch,(patchSize,patchSize),order=0)
            opatch[opatch != 0 ] = 1 
            opatch = opatch.astype(np.uint8)

            cords.append((i,startx,starty,endx,endy))


            mpatch[mpatch == 1] = 0
            mpatch[mpatch != 0] = 1

            mpatch = resizeImage(mpatch,(patchSize,patchSize),order=0)

            mpatch[mpatch != 0 ] = 1 
            mpatch = mpatch.astype(np.uint8)

            if ipatches is None: 
                ipatches = ipatch[None]
                opatches = opatch[None]
                mpatches = mpatch[None]
            else:
                ipatches = np.vstack((ipatches,ipatch[None]))
                opatches = np.vstack((opatches,opatch[None]))
                mpatches = np.vstack((mpatches,mpatch[None]))


            cnt = cnt + 1 

    return ipatches, opatches, mpatches, cords

if __name__ == "__main__":

    # the b-value set/ dataset for which the segmentations have to be saved. 
    bset = "cspca"

    outputfolder = fr"outputs\segmentations\{bset}"

    cases = [] 

    # collect case names from both the scans (test and retest) for the particular b-value set. 

    splitspathname = fr"{bset}_1_0"
    splitspath = fr"outputs\splits\{splitspathname}.json"
    splitsdict = DataUtil.readJson(splitspath)
    cases.extend(list(splitsdict.keys()))  

    splitspathname = fr"{bset}_2_0"
    splitspath = fr"outputs\splits\{splitspathname}.json"
    splitsdict = DataUtil.readJson(splitspath)
    cases.extend(list(splitsdict.keys()))  
    
    modality = 'ADC'

    # looping through the test retest scans and the cross validation loops
    for scan in range(1,3):
        for cv in range(3):

            print(scan,cv)

            # path to model checkpoint 
            modelpath = fr"modelcheckpoints\{bset}\{scan}_{cv}\checkpoint.pt"

            # initializing the trained model by loading the pre-traind weights. 
            model = UNet(n_channels=1,n_classes=2)
            model.load_state_dict(torch.load(modelpath, map_location=lambda storage, loc: storage))
            model.eval()

            newsize2D = 96 
            depth = 3 

            _min = 0 
            _max = 3000

            for j,name in enumerate(cases):

                dataset = name.split("_")[0]
                sb = Path(fr"..\Data\{dataset}\{modality}\1_Original_Organized_merged\{name}")

                name = sb.stem
                print(name,float(j)/len(cases))

                ret = getAugmentedData(sb,modality,nosamples=None)

                for k,aug in enumerate(ret):
            
                    augimg = aug[0]
                    augpm = aug[1][0]
                    augls = aug[1][1]
            
                    augpm_ = sitk.BinaryDilate(augpm,(2,2,0),sitk.sitkBall)

                    # Extracting patches         
                    ipatches, opatches, mpatches, cords = getPatches(augimg,augpm_,augls,_min,_max,newsize2D)

                    # Model output for the extracted patches. 
                    logits = model(torch.from_numpy(ipatches)) #[:,1,None,:,:]

                    logits = logits[:,1,:,:].detach().numpy()
                    ipatches = ipatches[:,0,:,:]

                    segmented = np.zeros((augimg.GetSize()[2],augimg.GetSize()[1],augimg.GetSize()[0]))

                    for samNo in range(len(cords)):
                        slno, startx,starty,endx,endy = cords[samNo]
                        xsize = endx - startx 
                        ysize = endy - starty 

                        slc = logits[samNo]
                        slc = np.array(Image.fromarray(slc).resize((ysize,xsize),Image.BILINEAR))

                        segmented[slno,startx:endx, starty:endy] = slc                         

                    DataUtil.mkdir(fr"{outputfolder}\{scan}_{cv}\{name}")


                    # Saving the segmented image, ADC image and the ground-truth.  
                    segmented = sitk.GetImageFromArray(segmented)
                    segmented = sitk.BinaryThreshold(segmented,0.5)

                    segmented = DataUtil.copyImageParameters(segmented,augimg)
                    augls = DataUtil.copyImageParameters(augls,augimg)
                    augpm = DataUtil.copyImageParameters(augpm,augimg)

                    sitk.WriteImage(augpm,fr"{outputfolder}\{scan}_{cv}\{name}\pm.nii.gz")
                    sitk.WriteImage(segmented,fr"{outputfolder}\{scan}_{cv}\{name}\prob.nii.gz")
                    sitk.WriteImage(augls,fr"{outputfolder}\{scan}_{cv}\{name}\gt.nii.gz")
                    sitk.WriteImage(augimg,fr"{outputfolder}\{scan}_{cv}\{name}\img.nii.gz")

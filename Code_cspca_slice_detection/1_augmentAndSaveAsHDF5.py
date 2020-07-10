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
from medImageProcessingUtil import MedImageProcessingUtil 
from skimage.measure import regionprops
from PIL import Image
from skimage.transform import resize as resizeImage

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


def createHDF5(splitspathname,splitsdict,patchSize):
    
    """
    splitspathname : dictionary containing filename vs their phases (train, test, val )
    splitsdict : splits dictionary. key : filename/case, value : phase (train,test)
    patchSize : x,y dimension of the image 
    """
    
    outputfolder = fr"outputs\hdf5\{splitspathname}"
    Path(outputfolder).mkdir(parents=True, exist_ok=True)

    img_dtype = tables.Float32Atom()
    ls_dtype = tables.UInt8Atom()
    pm_dtype = tables.UInt8Atom()
    data_shape = (0, patchSize, patchSize)
    mask_shape = (0,patchSize,patchSize)
    orgmask_shape = (0,patchSize,patchSize)

    filters = tables.Filters(complevel=5)

    phases = np.unique(list(splitsdict.values()))

    for phase in phases:
        hdf5_path = fr'{outputfolder}\{phase}.h5'

        if os.path.exists(hdf5_path):
            Path(hdf5_path).unlink()

        hdf5_file = tables.open_file(hdf5_path, mode='w')


        data = hdf5_file.create_earray(hdf5_file.root, "data", img_dtype,
                                            shape=data_shape,
                                            chunkshape = (1,patchSize,patchSize),
                                            filters = filters)

        mask =  hdf5_file.create_earray(hdf5_file.root, "mask", ls_dtype,
                                            shape=mask_shape,
                                            chunkshape = (1,patchSize,patchSize),
                                            filters = filters)

        orgmask =  hdf5_file.create_earray(hdf5_file.root, "orgmask", pm_dtype,
                                            shape=orgmask_shape,
                                            chunkshape = (1,patchSize,patchSize),
                                            filters = filters)

        hdf5_file.close()


def _addToHDF5(imgarr,maskarr,orgmaskarr,phase,splitspathname):
    
    """
    imgarr : input image sample (ex 2 slices of the image)
    maskarr : lesion mask (ex. lesion segmentation mask)
    orgmaskarr :  organ mask array
    phase : phase of that image (train,test,val)
    splitspathname : name of the file (json) which has train test splits info 
    """
    outputfolder = fr"outputs\hdf5\{splitspathname}"

    hdf5_file = tables.open_file(fr'{outputfolder}\{phase}.h5', mode='a')

    data = hdf5_file.root["data"]
    mask = hdf5_file.root["mask"]
    orgmask = hdf5_file.root["orgmask"]

    data.append(imgarr[None])
    mask.append(maskarr[None])
    orgmask.append(orgmaskarr[None])

    hdf5_file.close()

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


def addToHDF5(img,pm,ls,phase,splitspathname,_min,_max,patchSize):

    """ 
    Collect samples from the cropped volume and add them into HDF5 file 
    img : SimpleITK image to be pre-processed
    pm : prostate mask as SimpleITK image to be preprocesed
    ls : Lesion segmentation volume, a binary mask 
    phase : Train/test ? 
    splitspathname : The filename of hdf5 file 
    _min : Min value for the image to be normalized to
    _max : Max value for the image to be normalized to 
    """

    cnt = 0 

    # normalize the image between (0,1) give nthe min and max 
    imgarr = normalizeImage(img,_min,_max)

    pmarr = sitk.GetArrayFromImage(pm)
    lsarr = sitk.GetArrayFromImage(ls)

    size = imgarr.shape 

    # loop through each slice in the image 
    for i in range(1,size[0]-1):

        mask = lsarr[i]
        orgmask = pmarr[i]

        # consider only the slices which have have prostate voxel and leave out the ones outside prostate
        if orgmask.sum() != 0 :

            props = regionprops(orgmask)

            # bounding box of the prostate capsule 
            (startx,starty,endx,endy) = props[0].bbox

            mpatch = mask[startx:endx, starty:endy]
            ipatch = imgarr[i,startx:endx, starty:endy]
            opatch = orgmask[startx:endx, starty:endy]

            # resize image into the given patch size 
            ipatch = resizeImage(ipatch,(patchSize,patchSize))

            opatch = resizeImage(opatch,(patchSize,patchSize),order=0)
            opatch[opatch != 0 ] = 1 
            opatch = opatch.astype(np.uint8)

            # Mask all insignificant prostate cancer voxels. 
            mpatch[mpatch == 1] = 0
            mpatch[mpatch != 0] = 1

            mpatch = resizeImage(mpatch,(patchSize,patchSize),order=0)

            mpatch[mpatch != 0 ] = 1 
            mpatch = mpatch.astype(np.uint8)

            # add the image, lesion segmentation mask and the prostate capsule segmentation mask to hdf5
            _addToHDF5(ipatch,mpatch,opatch,phase,splitspathname)
            cnt = cnt + 1 

    return cnt 



def main():
   for scan in range(1,3):
        for cv in range(3):

            # modality of the imgaging data. The images saved with the same name. Ex. ADC.nii, T2W.nii
            modality = 'ADC'

            # no of augmentation samples to be generated 
            nosamples = 10

            # size of the final output patches
            newsize2D = 96 


            # path to the splits dictionary. Splits between train, test. key: Case/File name, value : Phase (either train/ test)
            splitspathname = fr"cspca_{scan}_{cv}"
            splitspath = fr"outputs\splits\{splitspathname}.json"
            splitsdict = DataUtil.readJson(splitspath)

            cases = list(splitsdict.keys())

            # generate an empty hdf5 file to store the patches. s
            createHDF5(splitspathname,splitsdict,newsize2D)

            casenames = {} 
            casenames["train"] = [] 
            casenames["val"] = [] 
            casenames["test"] = [] 

            # minimum and maximum value of the intensties in the image, to avoid image artifacts. 
            # The images are normalized with respect to this min and max. 
            _min = 0 
            _max = 3000


            # Read each volume, extract patches and store in hdf5 format. 
            for j,name in enumerate(cases):

                dataset = name.split("_")[0]
                sb = Path(fr"..\Data\{dataset}\{modality}\1_Original_Organized_merged\{name}")

                name = sb.stem
                print(name,float(j)/len(cases))

                phase = splitsdict[name]

                if phase == "train":
                    ret = getAugmentedData(sb,modality,nosamples=nosamples)
                else:
                    ret = getAugmentedData(sb,modality,nosamples=None)

                for k,aug in enumerate(ret):
            
                    augimg = aug[0]
                    augpm = aug[1][0]
                    augls = aug[1][1]

                    augpm = sitk.BinaryDilate(augpm,2,sitk.sitkBall)
                    
                    # Add the patches to the hdf5 file
                    cnt = addToHDF5(augimg,augpm,augls,phase,splitspathname,_min,_max,newsize2D)

                    # collect case/file names 
                    casename = name if k == 0 else fr"{name}_A{k}"
                    
                    for slno in range(cnt):
                        casenames[phase].append(fr"{casename}_{slno}")

            # saving the filenames, ground-truth information in the same hdf5 file 
            outputfolder = fr"outputs\hdf5\{splitspathname}"

            for phase in ["train","test","val"]:
                hdf5_file = tables.open_file(fr'{outputfolder}\{phase}.h5', mode='a')
                hdf5_file.create_array(hdf5_file.root, fr'names', casenames[phase])
                hdf5_file.close()




if __name__ == "__main__":
    main()
 
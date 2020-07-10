import sys 
sys.path.append(fr"..\Code_general")
from dataUtil import DataUtil 
import SimpleITK as sitk 
import numpy as np 
import pandas as pd 

import subprocess

def copy_parameters(img,ref):
    img.SetOrigin(ref.GetOrigin())
    img.SetDirection(ref.GetDirection())
    img.SetSpacing(ref.GetSpacing())

    return img 

def save_resampled_volumes(testcases,dataset,scan):

    for case in testcases:
        print(case)

        case1 = case 
        case2 = case.replace("Scan1","Scan2")

        probs1 = None 
        probs2 = None 
        for cv in range(3):

            spacing = (1,1,1)


            outputfolder1 = fr"outputs\segmentations\{dataset}\{scan}_{cv}\{case1}\reg"
            outputfolder2 = fr"outputs\segmentations\{dataset}\{scan}_{cv}\{case2}\reg"

            DataUtil.mkdir(outputfolder1)
            DataUtil.mkdir(outputfolder2)

            img1 = sitk.ReadImage(fr"outputs\segmentations\{dataset}\{scan}_{cv}\{case1}\img.nii.gz")
            img2 = sitk.ReadImage(fr"outputs\segmentations\{dataset}\{scan}_{cv}\{case2}\img.nii.gz")

            origin1 = img1.GetOrigin()
            origin2 = img2.GetOrigin()

            pm1 = sitk.ReadImage(fr"outputs\segmentations\{dataset}\{scan}_{cv}\{case1}\pm.nii.gz")
            pm2 = sitk.ReadImage(fr"outputs\segmentations\{dataset}\{scan}_{cv}\{case2}\pm.nii.gz")

            pm1 = copy_parameters(pm1,img1)
            pm2 = copy_parameters(pm2,img2)

            gt1 = sitk.ReadImage(fr"outputs\segmentations\{dataset}\{scan}_{cv}\{case1}\gt.nii.gz")
            gt2 = sitk.ReadImage(fr"outputs\segmentations\{dataset}\{scan}_{cv}\{case2}\gt.nii.gz")

            gt1 = copy_parameters(gt1,img1)
            gt2 = copy_parameters(gt2,img2)

            probs1 = sitk.ReadImage(fr"outputs\segmentations\{dataset}\{scan}_{cv}\{case1}\prob.nii.gz")
            probs2 = sitk.ReadImage(fr"outputs\segmentations\{dataset}\{scan}_{cv}\{case2}\prob.nii.gz")

            probs1 = copy_parameters(probs1,img1)
            probs2 = copy_parameters(probs2,img2)

            img1 = DataUtil.resampleimage(img1, spacing, origin1, interpolator = sitk.sitkLinear)
            gt1 = DataUtil.resampleimage(gt1, spacing, origin1, interpolator = sitk.sitkNearestNeighbor)
            pm1 = DataUtil.resampleimage(pm1, spacing, origin1, interpolator = sitk.sitkNearestNeighbor)
            probs1 = DataUtil.resampleimage(probs1, spacing, origin1, interpolator = sitk.sitkNearestNeighbor)
            img2 = DataUtil.resampleimage(img2, spacing, origin2, interpolator = sitk.sitkLinear)
            gt2 = DataUtil.resampleimage(gt2, spacing, origin2, interpolator = sitk.sitkNearestNeighbor)
            pm2 = DataUtil.resampleimage(pm2, spacing, origin2, interpolator = sitk.sitkNearestNeighbor)
            probs2 = DataUtil.resampleimage(probs2, spacing, origin2, interpolator = sitk.sitkNearestNeighbor)

            sitk.WriteImage(img1,fr"{outputfolder1}\img.nii.gz")
            sitk.WriteImage(gt1,fr"{outputfolder1}\gt.nii.gz")
            sitk.WriteImage(pm1,fr"{outputfolder1}\pm.nii.gz")
            sitk.WriteImage(probs1,fr"{outputfolder1}\prob.nii.gz")

            sitk.WriteImage(img2,fr"{outputfolder2}\img.nii.gz")
            sitk.WriteImage(gt2,fr"{outputfolder2}\gt.nii.gz")
            sitk.WriteImage(pm2,fr"{outputfolder2}\pm.nii.gz")
            sitk.WriteImage(probs2,fr"{outputfolder2}\prob.nii.gz")


def _save_registered_volumes(testcases,dataset,scan):

    for case in testcases:

        print(case)

        if scan == 1:
            case1 = case 
            case2 = case.replace("Scan1","Scan2")
        else:
            case2 = case 
            case1 = case.replace("Scan1","Scan2")     
        
        other_scan = 1 if scan == 2 else 1 

        probs1 = None 
        probs2 = None 


        for cv in range(3):

            outputfolder = fr"outputs\segmentations\{dataset}\{scan}_{cv}\{case1}\reg"

            segpath1 = fr"outputs\segmentations\{dataset}\{scan}_{cv}\{case1}\reg\img.nii.gz"
            segpath2 = fr"outputs\segmentations\{dataset}\{scan}_{cv}\{case2}\reg\img.nii.gz"

            pmpath1 = fr"outputs\segmentations\{dataset}\{scan}_{cv}\{case1}\reg\pm.nii.gz"
            pmpath2 = fr"outputs\segmentations\{dataset}\{scan}_{cv}\{case2}\reg\pm.nii.gz"

            gtpath1 = fr"outputs\segmentations\{dataset}\{scan}_{cv}\{case1}\reg\gt.nii.gz"
            gtpath2 = fr"outputs\segmentations\{dataset}\{scan}_{cv}\{case2}\reg\gt.nii.gz"
 
            probpath1 = fr"outputs\segmentations\{dataset}\{scan}_{cv}\{case1}\reg\prob.nii.gz"
            probpath2 = fr"outputs\segmentations\{dataset}\{scan}_{cv}\{case2}\reg\prob.nii.gz"

            cmd1 = "elastix -f {} -fMask {} -m {} -mMask {} -p affineTest.txt -out {}".format(segpath1,pmpath1,segpath2,pmpath2,outputfolder)
            subprocess.call(['cmd', '/c', cmd1])

            addline = "(FinalBSplineInterpolationOrder 0)"

            with open(fr"{outputfolder}\TransformParameters.0.txt","r") as infile:
                trans = infile.read()
            infile.close()

            trans = trans + "\n" + addline
            trans = trans.replace('(ResampleInterpolator "FinalLinearInterpolator")','(ResampleInterpolator "FinalNearestNeighborInterpolator")')

            with open(fr"{outputfolder}\TransformParametersMASK.0.txt","w") as infile:
                infile.writelines(trans)
            infile.close()


            DataUtil.mkdir(fr"{outputfolder}\img{other_scan}")
            DataUtil.mkdir(fr"{outputfolder}\gt{other_scan}")
            DataUtil.mkdir(fr"{outputfolder}\prob{other_scan}")

            cmd2 = "transformix -in {} -tp {}\\TransformParameters.0.txt -out {}\\img{}".format(segpath2,outputfolder,outputfolder,other_scan)
            subprocess.call(['cmd', '/c', cmd2])

            cmd2 = "transformix -in {} -tp {}\\TransformParametersMASK.0.txt -out {}\\gt{}".format(gtpath2,outputfolder,outputfolder,other_scan)
            subprocess.call(['cmd', '/c', cmd2])

            cmd3 = "transformix -in {} -tp {}\\TransformParametersMASK.0.txt -out {}\\prob{}".format(probpath2,outputfolder,outputfolder,other_scan)
            subprocess.call(['cmd', '/c', cmd3])


def save_registered_volumes(testcases,dataset):

    save_resampled_volumes(testcases,dataset,1)
    save_resampled_volumes(testcases,dataset,2)

    _save_registered_volumes(testcases,dataset,1)
    _save_registered_volumes(testcases,dataset,2)


if __name__ == "__main__":

    # the b-value set/ dataset for which the images have to be registered. 
    dataset = 'cspca'

    # collect file names for the dataset
    splitspathname = fr"{dataset}_1_0"
    splitspath = fr"outputs\splits\{splitspathname}.json"
    splitsdict = DataUtil.readJson(splitspath)
    samples = splitsdict.items()
    testcases = [x[0] for x in samples if x[1] == "test"] 

    save_registered_volumes(testcases,dataset)


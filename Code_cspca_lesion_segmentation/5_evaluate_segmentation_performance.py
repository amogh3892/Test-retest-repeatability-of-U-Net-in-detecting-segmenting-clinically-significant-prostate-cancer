import sys 
sys.path.append(fr"..\Code_general")
from dataUtil import DataUtil 
import SimpleITK as sitk 
import numpy as np 
import pandas as pd 
import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats
from skimage.measure import label as ConnectedComponent
from skimage.measure import regionprops
from rpy2.robjects import DataFrame, FloatVector, IntVector
from rpy2.robjects.packages import importr
from rpy2 import robjects as ro
from rpy2.robjects import pandas2ri
pandas2ri.activate()
import matplotlib.pyplot as plt 
from scipy.stats import ttest_ind,pearsonr,normaltest
# from pyCompare import blandAltman
from pingouin import plot_blandaltman


def filter_image(img,label):

    """
    Filter image given a label and mask everything else
    img: The image to be masked 
    label: The label in the image to be retained. 
    """

    img[img != label] = 0
    img[img > 0] = 1 
    img = img.astype(np.uint8)
    return img 

def removeSmallLesions(prob):
    """
    In order to reduce the false positives, we remove segmentation lesions which are less that 5 voxels in the major axis.
    prob: The segmented binary mask 
    """

    prob = sitk.GetArrayFromImage(prob)
    prob = ConnectedComponent(prob)

    prob_labels = np.unique(prob)
    prob_labels = prob_labels[prob_labels != 0].tolist()

    for p in prob_labels:
        prob_ = np.copy(prob)
        prob_[prob_ != p] = 0 

        props = regionprops(prob_)
        length = props[0].major_axis_length

        if length < 5:
            prob[prob == p] = 0 
            print("Removed")


    prob[prob > 1] = 1 
    prob = prob.astype(np.uint8)
    prob = sitk.GetImageFromArray(prob)

    return prob


def DiceLoss(input, target):

    """
    Calculate dice coefficient between two arrays
    """

    smooth = 1.
    iflat = input.reshape(-1)
    tflat = target.reshape(-1)
    intersection = (iflat * tflat).sum()
    
    if iflat.sum() + tflat.sum() == 0:
        return None 

    elif intersection == 0:
        return 0 

    return ((2.*intersection + smooth) /
              (iflat.sum() + tflat.sum() + smooth))



def removeFalsePositives(prob, gt):
    
    prob = sitk.GetArrayFromImage(prob)
    prob = ConnectedComponent(prob)

    gt = sitk.GetArrayFromImage(gt)


    prob_labels = np.unique(prob)
    prob_labels = prob_labels[prob_labels != 0].tolist()

    removed = 0 

    for p in prob_labels:
        prob_ = np.copy(prob)

        prob_[prob_ != p] = 0 
  
        gt_ = np.copy(gt)

        prob_[prob_ > 0] = 1
        gt_[gt_ > 0] = 1

        total = prob_.sum()
        overlap = np.multiply(prob_,gt_).sum()


        try:
            if not overlap/float(total) > 0.2:
                prob[prob == p] = 0 
                removed += 1 
                print("removed fp")
        except:
                import pdb 
                pdb.set_trace()



    prob[prob > 1] = 1 
    prob = prob.astype(np.uint8)
    prob = sitk.GetImageFromArray(prob)

    return prob,removed 

def filterSegmentation(prob):
    

    prob = sitk.GetArrayFromImage(prob)

    _prob = np.copy(prob)
    _prob[_prob > 0] = 1 

    cc = ConnectedComponent(_prob)

    prob_labels = np.unique(cc)
    prob_labels = prob_labels[prob_labels != 0].tolist()

    for p in prob_labels:
        cc_ = np.copy(cc)
        cc_[cc_ != p] = 0 
        cc_[cc_ > 0] = 1 

        check = (3 in np.multiply(prob,cc_)) or (2 in np.multiply(prob,cc_))
        # check = (3 in np.multiply(prob,cc_)) 


        if not check:
            cc[cc == p] = 0 
            print("Removed")


    cc[cc > 0] = 1

    cc = cc.astype(np.uint8)
    cc = sitk.GetImageFromArray(cc)

    return cc


def getIcc31(data1,data2):
    d1 = data1
    d2 = data2

    assert len(d1) == len(d2)

    d = np.column_stack((d1,d2))
    ddf = pd.DataFrame(d)
    rdf = pandas2ri.py2ri(ddf)

    irr = importr("irr")

    d = irr.icc(rdf,model='twoway',type='consistency',unit = "single",conf_level = 0.95)


    ci1 = d[13][0]
    ci2 = d[14][0]

    return (d[6][0], ci1, ci2)

def get_icc_dice(probs1,probs2):
    arr1 = sitk.GetArrayFromImage(probs1)
    arr2 = sitk.GetArrayFromImage(probs2)

    sl1 = set(np.unique(np.nonzero(arr1)[0]).flatten())
    sl2 = set(np.unique(np.nonzero(arr2)[0]).flatten())

    slnos = list(sl1.union(sl2))

    icc_dice = [] 

    for slno in slnos:
        slc1 = arr1[slno].flatten()
        slc2 = arr2[slno].flatten()

        icc,c1,c2 = getIcc31(slc1.tolist(),slc2.tolist())
        dice = DiceLoss(slc1, slc2)

        icc_dice.append((icc,dice))

    return icc_dice


def get_overlapped_lesion(prob,ref):
    co,no = ConnectedComponent(prob,return_num=True)

    dices = [] 

    ret = np.zeros(co.shape)

    for i in range(1,no+1):
        tempco = np.copy(co)
        tempco[tempco != i] = 0
        tempco[tempco == i] = 1

        dice = DiceLoss(ref,tempco)
        assert tempco.max() <= 1
        

        if dice > 0.2:
            ret = ret + tempco

    ret[ret >  1] = 1 
    return ret 



def get_dice_repeatability(gt1,probs1):

    garry = sitk.GetArrayFromImage(gt1)

    
    hits = 0 
    misses = 0 
    fps = 0 

    gt1,misses = removeFalsePositives(gt1,probs1)
    probs1,fps = removeFalsePositives(probs1, gt1)

    dices = None

    g1 = sitk.GetArrayFromImage(gt1)
    p1 = sitk.GetArrayFromImage(probs1)

    c1,n1 = ConnectedComponent(g1,return_num=True)
    c3,n3 = ConnectedComponent(p1,return_num=True)

    print(n1,n3)
    # assert n1 == n2 == n3 == n4 

    if (n1 > 0):
        dices = []

        for i in range(1,n1+1):
            c1_ = np.copy(c1)
            c1_[c1_ != i] = 0
            c1_[c1_ == i] = 1

            if c1_.max() > 1:
                import pdb 
                pdb.set_trace()

            c3_ = get_overlapped_lesion(p1,c1_)

            dice1 = DiceLoss(c1_, c3_)

            dices.append(dice1)
            print(dices)

        
        hits = len(dices)

    return dices, hits, misses, fps   



def evaluate_segmentation_performance_repeatability_holdout(testcases,dataset):

    """
    Evaluates lesion detection and segmentation performance of the cross validation set. 
    Also evaluates repeatability of lesion segmentation in terms of dice similarity coefficient

    testcases : filenames of testcases 
    dataset: b-value settings or the dataset name

    returns: a tuple (Mean and standard deviation of dice of first scan, 
                        Mean, standard deviation of dice of second scan,
                        Mean, standard deviation of dice between scans (repeatability)
                        # hits (No of lesions detected), # misses, # false positives for scan1,
                        # # hits (No of lesions detected), # misses, # false positives for scan2,
                        # agreement and disagreements between the network.)

    """


    dices = []
    h1 = 0
    h2 = 0 
    h3 = 0 

    f1 = 0 
    f2 = 0 
    f3 = 0 

    m1 = 0 
    m2 = 0 
    m3 = 0

    for case in testcases:
    
        probs1 = None 
        probs2 = None 

        gtpath1 = fr"outputs\segmentations\{dataset}\1_0\{case}\gt.nii.gz"
        gt1 = sitk.ReadImage(gtpath1)
        gt1 = sitk.GetArrayFromImage(gt1)
        gt1[gt1 == 1] = 0 
        gt1 = sitk.GetImageFromArray(gt1)
        gt1 = DataUtil.convert2binary(gt1)

        for cv in range(3):

            probpath1 = fr"outputs\segmentations\{dataset}\1_{cv}\{case}\prob.nii.gz"
            probpath2 = fr"outputs\segmentations\{dataset}\2_{cv}\{case}\prob.nii.gz"

            probs1_ = DataUtil.convert2binary(sitk.ReadImage(probpath1))
            probs2_ = DataUtil.convert2binary(sitk.ReadImage(probpath2))

            probs1 = probs1_ if probs1 is None else sitk.Add(probs1, probs1_)
            probs2 = probs2_ if probs2 is None else sitk.Add(probs2, probs2_)


        probs1 = filterSegmentation(probs1)
        probs2 = filterSegmentation(probs2)

        probs1 = removeSmallLesions(probs1)
        probs2 = removeSmallLesions(probs2)

        dice1, hits1, misses1, fps1 = get_dice_repeatability(gt1,probs1)
        dice2, hits2, misses2, fps2 = get_dice_repeatability(gt1,probs2)
        dice3, hits3, misses3, fps3 = get_dice_repeatability(probs1,probs2)

        dices.append((dice1,dice2,dice3))

        h1 += hits1 
        m1 += misses1 
        f1 += fps1 

        h2 += hits2
        m2 += misses2
        f2 += fps2

        h3 += hits3
        m3 += misses3
        f3 += fps3

    dice1,dice2,dice3 = zip(*dices)


    dice1 = [y for x in dice1 if x is not None for y in x ]
    dice2 = [y for x in dice2 if x is not None for y in x ]
    dice3 = [y for x in dice3 if x is not None for y in x ]

    print(np.mean(dice1),np.std(dice1))
    print(np.mean(dice2),np.std(dice2))
    print(np.mean(dice3),np.std(dice3))

    print(h1,m1,f1)
    print(h2,m2,f2)
    print(h3,m3,f3)

    return ((np.mean(dice1),np.std(dice1)), (np.mean(dice2),np.std(dice2)), (np.mean(dice3),np.std(dice3)),(h1,m1,f1),(h2,m2,f2),(h3,m3+f3))


def evaluate_segmentation_cv(scan,dataset):
    
    """
    Evaluates lesion detection and segmentation performance of the cross validation set. 
    scan : Test/ Retest 
    dataset: b-value settings or the dataset name

    returns: a tuple (Mean segmentation dice score, standard deviation of segmentation dice, 
                        # hits (No of lesions detected), # misses, # false positives)

    
    """

    dices = [] 

    h1= 0 
    m1 = 0 
    f1 = 0 

    for cv in range(3):

        segpath = fr"outputs\segmentations\{dataset}\{scan}_{cv}"

        splitspathname = fr"{dataset}_{scan}_{cv}"
        
        splitspath = fr"outputs\splits\{splitspathname}.json"
        splitsdict = DataUtil.readJson(splitspath)
        samples = splitsdict.items()

        valcases = [x[0] for x in samples if x[1] == "val"] 

        for case in valcases:
            probs = sitk.ReadImage(fr"{segpath}\{case}\prob.nii.gz")
            probs = DataUtil.convert2binary(probs)
            probs = removeSmallLesions(probs)

            gt = sitk.ReadImage(fr"{segpath}\{case}\gt.nii.gz")

            gt = sitk.GetArrayFromImage(gt)
            gt[gt == 1] = 0 
            gt = sitk.GetImageFromArray(gt)
            gt = DataUtil.convert2binary(gt)

            dice1, hits1, misses1, fps1 = get_dice_repeatability(gt,probs)
            dices.append(dice1)

            h1 += hits1 
            m1 += misses1 
            f1 += fps1 

    dices = [y for x in dices if x is not None for y in x]

    return (np.mean(dices), np.std(dices), h1,m1,f1)




if __name__ == "__main__":
    
    dataset = 'cspca'

    splitspathname = fr"{dataset}_1_0"
    splitspath = fr"outputs\splits\{splitspathname}.json"
    splitsdict = DataUtil.readJson(splitspath)
    samples = splitsdict.items()
    test1 = [x[0] for x in samples if x[1] == "test"] 

    splitspathname = fr"{dataset}_2_0"
    splitspath = fr"outputs\splits\{splitspathname}.json"
    splitsdict = DataUtil.readJson(splitspath)
    samples = splitsdict.items()
    test2 = [x[0] for x in samples if x[1] == "test"] 

    testcases = []
    testcases.extend(test1)
    testcases.extend(test2)

    d1,s1,h1,m1,f1  = evaluate_segmentation_cv(1,dataset)
    d2,s2,h2,m2,f2  = evaluate_segmentation_cv(2,dataset)


    evaluate_segmentation_performance_repeatability_holdout(testcases, dataset)



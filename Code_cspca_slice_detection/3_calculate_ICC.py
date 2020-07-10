from rpy2.robjects import DataFrame, FloatVector, IntVector
from rpy2.robjects.packages import importr
from rpy2 import robjects as ro
from rpy2.robjects import pandas2ri
pandas2ri.activate()
from sklearn.metrics import roc_auc_score 

from sklearn.decomposition import PCA

from collections import namedtuple
import numpy as np
import pandas as pd 
import pickle
from sklearn.metrics import roc_auc_score,roc_curve,auc

import matplotlib.pyplot as plt 
plt.rcParams.update({'font.size': 20})
import matplotlib.animation as animation
import os 

import scipy.stats

import sys 
sys.path.append(fr"..\Code_general")
from dataUtil import DataUtil 

def mean_confidence_interval(data, confidence=0.95):
    """
    95% confidence interval for the input data.
    """

    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h



def getIcc31(data1,data2):

    """
    data1 : First set of points
    data2 : Second set of points
    Get icc(3,1) for the two distributions with confidence internval
    
    """

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


def getAUC(ytrue,ypred):

    """
    ytrue : Ground-truth values 
    ypred : Predictions by the network/ ML algorithm 
    AUC with condiference intervals
    
    """

    ytrue = np.concatenate((ytrue, ytrue), axis=0)
    ypred = np.concatenate((ypred, ypred), axis=0)


    rpackage_pROC = importr('pROC')
    rpackage_base = importr('base')
    rpackage_psych = importr('psych')


    #  2000 stratified bootstrap replicates
    r_rocobj = rpackage_pROC.roc(ytrue, ypred)

    AUC = r_rocobj[8][0]

    r_ci = rpackage_pROC.ci(r_rocobj, x="best")

    
    return r_ci[1],r_ci[0],r_ci[2]



def populateCrossValAUC(models,inputfolder,outputfilename,cvsplits=3):

    output = [] 
    columns = []

    for i,model in enumerate(models):
        row = []
        
        pred1 = [] 
        true1 = [] 
        pred2 = [] 
        true2 = [] 

        for cv in range(cvsplits):

            df1 = pd.read_csv(fr"{inputfolder}\{model}_1_{cv}\predictions.csv",index_col=0)
            df2 = pd.read_csv(fr"{inputfolder}\{model}_2_{cv}\predictions.csv",index_col=0)

            df1 = df1[df1["Phase"]=="val"]
            df2 = df2[df2["Phase"]=="val"]

            pred1.extend(df1["Pred"].values)
            true1.extend(df1["True"].values)

            pred2.extend(df2["Pred"].values)
            true2.extend(df2["True"].values)


        a1,c11,c12 = getAUC(true1,pred1)
        a2,c21,c22 = getAUC(true2,pred2)

        row.extend([model,a1,c11, c12,a2,c21,c22])
        if i == 0:
            columns.extend([fr"Model",fr"AUC-Scan1",fr"CI1-Scan1",fr"CI2-Scan1",fr"AUC-Scan2",fr"CI1-Scan2",fr"CI2-Scan2",])

        output.append(tuple(row))


    df = pd.DataFrame(output,columns=columns)

    DataUtil.mkdir(fr"outputs\results")

    df.to_csv(fr"outputs\results\{outputfilename}_cv.csv",index=None)


def populateTestResults(models,inputfolder,outputfilename,cvsplits=3):
    output = [] 
     
    for model in models:
        df = None
        for cv in range(cvsplits):
            

            df1 = pd.read_csv(fr"{inputfolder}\{model}_1_{cv}\predictions.csv",index_col=0)
            df2 = pd.read_csv(fr"{inputfolder}\{model}_2_{cv}\predictions.csv",index_col=0)

            df1 = df1[df1["Phase"]!="val"]
            df2 = df2[df2["Phase"]!="val"]

            df1 = df1.rename(columns={'Pred': fr'Pred_{cv}'})
            df2 = df2.rename(columns={'Pred': fr'Pred_{cv}'})

            dfcv = df1.merge(df2, on=["FileName","True","Phase"])

            df = dfcv if df is None else df.merge(dfcv,on=["FileName","True","Phase"])

        df["Pred_x"] = df["Pred_0_x"] + df["Pred_1_x"] + df["Pred_2_x"]  
        df["Pred_y"] = df["Pred_0_y"] + df["Pred_1_y"] + df["Pred_2_y"]  

        ppred1 = df[(df["Phase"]=="test1")|(df["Phase"]=="test2")]["Pred_x"].values
        ppred2 = df[(df["Phase"]=="test1")|(df["Phase"]=="test2")]["Pred_y"].values

        ppred1_auc = df[(df["Phase"]=="test1")]["Pred_x"].values
        ppred2_auc = df[(df["Phase"]=="test2")]["Pred_y"].values

        ptrue1 = df[(df["Phase"]=="test1")]["True"].values
        ptrue2 = df[(df["Phase"]=="test2")]["True"].values

        picc31,pci1,pci2 = getIcc31(ppred1,ppred2)
        p1auc,p1c1,p1c2 = getAUC(ptrue1,ppred1_auc)
        p2auc,p2c1,p2c2 = getAUC(ptrue2,ppred2_auc)

        output.append((model,p1auc,p1c1,p1c2,p2auc,p2c1,p2c2,picc31,pci1,pci2))


    df = pd.DataFrame(output,columns=["Model","AUC1","AUC1-CI1","AUC1-CI2","AUC2","AUC2-CI1","AUC2-CI2","ICC","ICC-CI1","ICC-CI2"])

    DataUtil.mkdir(fr"outputs\results")
    df.to_csv(fr"outputs\results\{outputfilename}_holdout.csv",index=None)


if __name__ == '__main__':

    # Filename/dataset from which the performance metrics to be calculated and saved
    outputfilename = fr"cspca"

    models = [fr"cspca\unet"]

    # path where the model check points are saved.
    inputfolder = fr"modelcheckpoints"

    populateTestResults(models,inputfolder,outputfilename)
    populateCrossValAUC(models,inputfolder,outputfilename)

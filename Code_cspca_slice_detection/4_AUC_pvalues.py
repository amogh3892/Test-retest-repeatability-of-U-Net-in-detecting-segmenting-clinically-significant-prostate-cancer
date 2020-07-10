import pandas as pd
import sys 
sys.path.append(fr"..\Code_general")
from dataUtil import DataUtil 
from machineLearningUtil import PerformanceMetrics
import itertools

def mainExperimentCV(inputfolder,model,cvsplits=3):
    
    """
    inputfolder : path to model checkpoints. 
    model : name of the model. 
    cvsplits : k- fold cross validation.
    """

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

    pm1 = PerformanceMetrics(true1,pred1)
    pm2 = PerformanceMetrics(true2,pred2)

    pvalue = pm1.compare_auc(pm2)

    return pvalue 




def mainExperimentHoldOut(inputfolder,model,cvsplits):

    """
    inputfolder : path to model checkpoints. 
    model : name of the model. 
    cvsplits : k- fold cross validation.
    """

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

    ppred1_auc = df[(df["Phase"]=="test1")]["Pred_x"].values
    ppred2_auc = df[(df["Phase"]=="test2")]["Pred_y"].values

    ptrue1 = df[(df["Phase"]=="test1")]["True"].values
    ptrue2 = df[(df["Phase"]=="test2")]["True"].values

    pm1 = PerformanceMetrics(ptrue1,ppred1_auc)
    pm2 = PerformanceMetrics(ptrue2,ppred2_auc)

    pvalue = pm1.compare_auc(pm2)

    return pvalue 




if __name__ == "__main__":

    ### p-values for the main experiment (on the hold out test set)
    model = fr"cspca\unet"
    inputfolderMN = fr"..\modelcheckpoints"
    holdout_pvalue = mainExperimentHoldOut(inputfolderMN,model,cvsplits=3)
    cv_pvalue = mainExperimentCV(inputfolderMN,model,cvsplits=3)
  
    print(holdout_pvalue)
    print(cv_pvalue)

    import pdb 
    pdb.set_trace()



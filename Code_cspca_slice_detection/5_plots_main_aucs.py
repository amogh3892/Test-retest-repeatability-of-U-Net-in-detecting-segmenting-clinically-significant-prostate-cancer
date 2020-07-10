import sys 
sys.path.append(fr"..\Code_general")
from dataUtil import DataUtil 
from plotUtil import PlotUtil
import pandas as pd 


def plotCrossValidationROCurves(inputfolder,model,outputfilename,cvsplits=3,pvalue=None):

    """
    inputfolder : inputfolder where checkpoints are present
    model : The intialized model for loading the pretrained weights
    outputfilename : The output filename 
    pvalue : pvalue between ROC curves of cross validation test (Scan1) and retest (Scan2).    
    
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

        pred1.append(df1["Pred"].values)
        true1.append(df1["True"].values)

        pred2.append(df2["Pred"].values)
        true2.append(df2["True"].values)

    DataUtil.mkdir(fr"outputs\results")

    plt = PlotUtil.plotMultipleCrossValidationROCurves([true1,true2],[pred1,pred2],[r"$A_{m}$",r"$B_{m}$"],pvalue=pvalue)


    plt.savefig(fr'outputs\results\{outputfilename}.png',  
                bbox_inches = 'tight',
                pad_inches = 0)



def plotHoldOutTestROCurves(inputfolder,model,outputfilename,cvsplits=3,pvalue=None):

    """
    inputfolder : inputfolder where checkpoints are present
    model : Model name 
    outputfilename : The output filename 
    pvalue : pvalue between ROC curves of cross validation test (Scan1) and retest (Scan2).    
    
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



    pred1 = df[(df["Phase"]=="test1")]["Pred_x"].values
    pred2 = df[(df["Phase"]=="test2")]["Pred_y"].values

    true1 = df[(df["Phase"]=="test1")]["True"].values
    true2 = df[(df["Phase"]=="test2")]["True"].values


    DataUtil.mkdir(fr"outputs\results")

    plt = PlotUtil.plotROCurve([true1,true2],[pred1,pred2],[fr"$C_A$",fr"$C_B$"],pvalue=pvalue)

    plt.savefig(fr'outputs\results\{outputfilename}.png',  
                bbox_inches = 'tight',
                pad_inches = 0)



if __name__ == "__main__":


    # The dataset/filename or the model for which ROC plots to be saved. 
    model = fr"cspca\unet"

    # patch to the modelcheckpoints
    inputfolderMN = fr"..\modelcheckpoints"


    # Input the previously calculated p-values 
    plotCrossValidationROCurves(inputfolderMN,model,"CV_ROC_ResNet_cspca",cvsplits=3,pvalue=0.64)
    plotHoldOutTestROCurves(inputfolderMN,model,"HOLDOUT_ROC_ResNet_cspca",cvsplits=3,pvalue=0.58)



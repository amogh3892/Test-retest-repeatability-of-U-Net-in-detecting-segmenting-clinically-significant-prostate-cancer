
import numpy as np 
from rpy2.robjects import DataFrame, FloatVector, IntVector, StrVector
from rpy2.robjects.packages import importr
from rpy2 import robjects as ro
from rpy2.robjects import pandas2ri
pandas2ri.activate()
from sklearn.metrics import roc_auc_score 
rpackage_pROC = importr('pROC')
rpackage_base = importr('base')
rpackage_psych = importr('psych')

class PerformanceMetrics(object):
    def __init__(self,ytrue,ypred,cutoff=None,cutoffvariable="threshold"):

        self.ytrue = ytrue 
        self.ypred = ypred 

        self.aucobj = self._get_auc_obj() 
        self.auc = self.aucobj[8][0]
        self.ci = rpackage_pROC.ci(self.aucobj, x="best")
        self.lowci = self.ci[0]
        self.highci = self.ci[1]
        self.binary = None 

        if cutoff is None:
            cutoffmetrics = rpackage_pROC.coords(self.aucobj,"best", cutoffvariable,ret=StrVector(["threshold","specificity", "sensitivity","accuracy","ppv","npv"]))
        
        else:
            cutoffmetrics = rpackage_pROC.coords(self.aucobj,cutoff, cutoffvariable,ret=StrVector(["threshold","specificity", "sensitivity","accuracy","ppv","npv"]))

            if cutoffvariable == "threshold":
                binary_pred = [1 if x > cutoff else 0 for x in ypred]
                self.binary = binary_pred

        self.threshold = cutoffmetrics[0]
        self.specificity = cutoffmetrics[1]
        self.sensitivity = cutoffmetrics[2]
        self.accuracy = cutoffmetrics[3]
        self.ppv = cutoffmetrics[4]
        self.npv = cutoffmetrics[5]

    def _get_auc_obj(self):
        ytrue = self.ytrue 
        ypred = self.ypred

        ytrue = np.concatenate((ytrue, ytrue), axis=0)
        ypred = np.concatenate((ypred, ypred), axis=0)

        rpackage_pROC = importr('pROC')
        rpackage_base = importr('base')
        rpackage_psych = importr('psych')


        #  2000 stratified bootstrap replicates
        r_rocobj = rpackage_pROC.roc(ytrue, ypred)

        return r_rocobj

    def compare_auc(self,obj):

        t1 = rpackage_pROC.roc_test(self.aucobj,obj.aucobj)
        pvalue = t1[7][0]

        return pvalue

if __name__ == "__main__":

    ytrue = [1,0,1,0,1,0,1,0,1,0,1,0]
    ypred = [0.9,0.2,0.8,0.3,0.7,0.1,0.88,0.4,1,0.3,0.99,0]

    pm = PerformanceMetrics(ytrue,ypred)

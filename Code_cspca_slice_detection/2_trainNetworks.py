import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.models as models
import h5py
from torch.autograd import Variable
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt 
from pytorchtools import EarlyStopping
from random import randint
from PIL import Image
from sklearn.metrics import roc_auc_score 
from segUtil import *
import pandas as pd 
import tables

class ProstateDatasetHDF5(Dataset):
    def __init__(self, fname,transforms = None):
        self.fname=fname
        self.file = tables.open_file(fname)
        self.tables = self.file.root
        self.nitems=self.tables.data.shape[0]
        self.file.close()
        self.data = None
        self.mask = None
        self.names = None
        self.transforms = transforms
         
    def __getitem__(self, index):
                
        self.file = tables.open_file(self.fname)
        self.tables = self.file.root
        self.data = self.tables.data
        self.mask = self.tables.mask
        self.orgmask = self.tables.orgmask
        
        if "names" in self.tables:
            self.names = self.tables.names

        img = self.data[index,:,:]
        mask = self.mask[index,:,:]
        orgmask = self.orgmask[index,:,:]
        
        img = img*orgmask
                
        if self.names is not None:
            name = self.names[index]
            
        self.file.close()

        if mask.sum() == 0:
            label = 0 
        else:
            label = 1 
            

        return img[None],(label,name,mask)

    def __len__(self):
        return self.nitems


def getData(batch_size, num_workers,modelname, modelsizes,dataset,scan,cv):
    
    # The path to the hdf5 files (train, val and test of Scan1 and Scan2)
    trainfilename = fr"/mnt/data/home/axh672/{dataset}/{dataset}_{scan}_{cv}/train.h5"
    valfilename = fr"/mnt/data/home/axh672/{dataset}/{dataset}_{scan}_{cv}/val.h5"
    test1filename = fr"/mnt/data/home/axh672/{dataset}/{dataset}_1_{cv}/test.h5"
    test2filename = fr"/mnt/data/home/axh672/{dataset}/{dataset}_2_{cv}/test.h5" 

    train = h5py.File(trainfilename,libver='latest',mode='r')
    val = h5py.File(valfilename,libver='latest',mode='r')
    test1 = h5py.File(valfilename,libver='latest',mode='r')
    test2 = h5py.File(valfilename,libver='latest',mode='r')

    trainnames = np.array(train["names"])
    valnames = np.array(val["names"])
    test1names = np.array(test1["names"])
    test2names = np.array(test2["names"])

    train.close()
    val.close()
    test1.close()
    test2.close()
    

    # Defining Dataset class 
    data_train = ProstateDatasetHDF5(trainfilename,transforms=None)
    data_val = ProstateDatasetHDF5(valfilename,transforms=None)
    data_test1  = ProstateDatasetHDF5(test1filename,transforms=None)
    data_test2  = ProstateDatasetHDF5(test2filename,transforms=None)

    # Defining DataLoader class 
    trainLoader = torch.utils.data.DataLoader(dataset=data_train,batch_size = batch_size,num_workers = num_workers,shuffle = True)
    valLoader = torch.utils.data.DataLoader(dataset=data_val,batch_size = batch_size,num_workers = num_workers,shuffle = False) 
    test1Loader = torch.utils.data.DataLoader(dataset=data_test1,batch_size = batch_size,num_workers = num_workers,shuffle = False) 
    test2Loader = torch.utils.data.DataLoader(dataset=data_test2,batch_size = batch_size,num_workers = num_workers,shuffle = False) 

    dataLoader = {}
    dataLoader['train'] = trainLoader
    dataLoader['val'] = valLoader
    dataLoader['test1'] = test1Loader
    dataLoader['test2'] = test2Loader

    return dataLoader 



def run(mn, num_epochs, learning_rate, weightdecay, patience, scan, cv):

    # other network models can be added. 

    if mn == "unet":
        model = UNetDet(1,2)
    elif mn == "vggnet11":
        model = models.vgg11()
        model.classifier[6]  = nn.Linear(4096, 2)
    elif mn == "resnet18":
        model = models.resnet18()
        model.fc = nn.Linear(512, 2)


    # The GPU device number in which the program has to be executed. 
    device = torch.device("cuda:1")
    model.to(device)

    # Loss criteria and the optimization function
    criterion=nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weightdecay)

    # Defining the earlystopping criteria 
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    modelname = r"{}_{}_{}".format(mn,scan,cv)
    print(modelname)

    niter_total=len(dataLoader['train'].dataset)/batch_size

    # Looping through epochs
    for epoch in range(num_epochs):
        
        pred_df_dict = {} 
        results_dict = {} 
        
        for phase in ["train","test1","test2","val"]:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            confusion_matrix=np.zeros((2,2))
            ytrue = [] 
            ypred = [] 
            ynames = [] 
            loss_vector=[]
            
            # Looping through phases (train,val,test)
            for ii,(data,info) in enumerate(dataLoader[phase]):
                

                label = info[0]
                name = info[1]
                
                label=label.squeeze().long().to(device)
                data = Variable(data.float().cuda(device))
                
                with torch.set_grad_enabled(phase == 'train'):
                    
                    # get model output 
                    output = model(data).squeeze()
                    
                    _,pred_label=torch.max(output,1)
                    probs = F.softmax(output,dim = 1)
                    
                    # calculate loss 
                    loss = criterion(probs, label)

                    # output probabilities 
                    probs = probs[:,1]
                    
                    loss_vector.append(loss.detach().data.cpu().numpy())

                    # generate backward gradients, change weights based on optimization 
                    if phase=="train":
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()     
                    
                    
                    ypred.extend(probs.cpu().data.numpy().tolist())
                    ytrue.extend(label.cpu().data.numpy().tolist())
                    ynames.extend(list(name))

                    
                    pred_label=pred_label.cpu()
                    label=label.cpu()

                    for p,l in zip(pred_label,label):
                        confusion_matrix[p,l]+=1

            # performance metrics 
            total=confusion_matrix.sum()        
            acc=confusion_matrix.trace()/total
            loss_avg=np.mean(loss_vector)
            auc = roc_auc_score(ytrue,ypred)
                            
            # creating a dataframe to store the probability outputs of the network 
            if phase != "train": 
                pred_df = pd.DataFrame(np.column_stack((ynames,ytrue,ypred,[phase]*len(ynames))), 
                                    columns = ["FileName","True", "Pred","Phase"])
                pred_df_dict[phase] = pred_df

            # results dictionary of results 
            results_dict[phase] = {} 
            results_dict[phase]["loss"] = loss_avg
            results_dict[phase]["auc"] = auc 
            results_dict[phase]["acc"] = acc 

            if phase == 'train':
                print("Epoch : {}, Phase : {}, Loss : {}, Acc: {}, Auc : {}".format(epoch,phase,loss_avg,acc,auc))
            else:
                print("                 Epoch : {}, Phase : {}, Loss : {}, Acc: {}, Auc : {}".format(epoch,phase,loss_avg,acc,auc))
                

            for cl in range(confusion_matrix.shape[0]):
                cl_tp=confusion_matrix[cl,cl]/confusion_matrix[:,cl].sum()
                
            
            if phase == "val":
                df = pred_df_dict["val"].append(pred_df_dict["test1"], ignore_index=True).append(pred_df_dict["test2"], ignore_index=True)
                early_stopping(loss_avg, model, modelname,df, results_dict, parentfolder = None)

            if early_stopping.early_stop:
                print("Early stopping")
                break

        if early_stopping.early_stop:
            break



if __name__ == "__main__":

    # Loopoing through Scans 1 and 2 test and re-test
    scans = [1,2]

    # Looping through a three fold cross validation loop 
    cvs = [0,1,2]

    modelname = "unet"
    dataset = "cspca"

    batch_sizes = [64]

    # number of workers for parallel processing
    num_workers = 8

    # maximum number of epochs. An early stopping criteria is used to stop the network based on validation loss
    # If the validation loss increases consecutively > patience times, state of the model before increase is saved. 
    num_epochs = 200
    patience = 20 

    # Learning rate and weight decay parameters for Adam optimizer 
    learning_rate = 1e-5
    weightdecay = 1e-6
    
    # A dictionary for modelname vs sizes (when multiple architecture of networks are to be run)
    modelsizes = {} 
    modelsizes["unet"] = 96

    for scan in scans:
        for cv in cvs:
            for batch_size in batch_sizes:
                dataLoader = getData(batch_size,num_workers,modelname,modelsizes,dataset,scan,cv)
                run(modelname, num_epochs, learning_rate, weightdecay, patience, scan, cv)

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
from segUtil import UNet,ProstateDatasetHDF5
import tables

def DiceLoss(input, target):
    smooth = 1.
    
    iflat = input.reshape(-1)
        
    tflat = target.reshape(-1)
    intersection = (iflat * tflat).sum()
    
    return 1 - ((2. * intersection + smooth) /
              (iflat.sum() + tflat.sum() + smooth))


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
                                
        if self.names is not None:
            name = self.names[index]
        
        self.file.close()
        
        img = torch.from_numpy(img)
        mask = torch.from_numpy(mask)
        
        return img[None],mask,name

    def __len__(self):
        return self.nitems


def getData(batch_size, num_workers,size,dataset,scan,cv):

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
    
    data_train = ProstateDatasetHDF5(trainfilename,transforms=None)
    data_val = ProstateDatasetHDF5(valfilename,transforms=None)
    data_test1  = ProstateDatasetHDF5(test1filename,transforms=None)
    data_test2  = ProstateDatasetHDF5(test2filename,transforms=None)

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


def run(num_epochs, learning_rate, batch_size, weightdecay, patience,dataset, scan, cv ):

    # initializing the model 
    model = UNet(n_channels=1,n_classes=2)

    # load pre-trained model traind for slice- level clinically significant prostate cancer detection. 
    pretrained = torch.load(fr"Data/modelcheckpoints/cspca_modelcheckpoints/{dataset}/unet_{scan}_{cv}/checkpoint.pt")

    for item in pretrained.keys():
        if item in model.state_dict().keys():
            splits = item.split(".")
            number = int(splits[-2])
            model.state_dict()[item].data.copy_(pretrained[item])

    # The GPU device number in which the program has to be executed. 
    device = torch.device("cuda:1")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weightdecay)

    early_stopping = EarlyStopping(patience=patience, verbose=True)

    modelname = fr"{dataset}/{scan}_{cv}"
    print(modelname)

    niter_total=len(dataLoader['train'].dataset)/batch_size

    # Looping through epochs
    for epoch in range(num_epochs):
        
        for phase in ["train","val"]:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode


            loss_vector=[]

            # Looping through phases (train,val,test)
            for ii,(data,mask,name) in enumerate(dataLoader[phase]):
            
                data = Variable(data.float().cuda(device))
                mask = Variable(mask.float().cuda(device))
                
                logits = model(data)
                logits = logits[:,1,:,:]

                loss = DiceLoss(logits, mask)
        
                loss_vector.append(loss.detach().data.cpu().numpy())

                if phase=="train":
                    optimizer.zero_grad()

                    loss.backward()
                    optimizer.step()  

            loss_avg=np.mean(loss_vector)
            
            if phase == 'train':
                print("Epoch : {}, Phase : {}, Loss : {}".format(epoch,phase,loss_avg))
            else:
                print("                 Epoch : {}, Phase : {}, Loss : {}".format(epoch,phase,loss_avg))

                
            if phase == 'val':
                early_stopping(loss_avg, model, modelname,parentfolder = None)

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
    
    batch_size = 64

    # number of workers for parallel processing
    num_workers = 8

    size = 96 

    # maximum number of epochs. An early stopping criteria is used to stop the network based on validation loss
    # If the validation loss increases consecutively > patience times, state of the model before increase is saved. 
    num_epochs = 200
    patience = 10 

    
    # Learning rate and weight decay parameters for Adam optimizer 
    learning_rate = 1e-4
    weightdecay = 1e-6
    
    
    dataset = "cspca"

    for cv in cvs:
        for scan in scans:
            dataLoader = getData(batch_size,num_workers,size, dataset,scan,cv)
            run(num_epochs, learning_rate, batch_size, weightdecay, patience,dataset, scan, cv)

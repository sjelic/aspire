# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 01:43:34 2020

@author: sjelic
"""

import torch
from torch import nn


# class Conv2DExt(nn.Conv2d):
#     def __init__(self,in_channels, out_channels, kernel_size):
#         super().__init__(self, in_channels, out_channels, kernel_size)
#     def forward(self, input, output_size=None):
#         if input.ndimensions() == 5:
#             BI, FI, CI, HI, WI = input.shape
#             input = input.view(-1, CI, HI, WI)
#             output = super().forward(input, output_size=output_size)
#             BFO, CO, HO, WO = output.shape
#             output=output.view(BI, FI, CO, HO, WO)
#         else:
#             output = super().forward(input, output_size=output_size)
             
            
#         return output

# class Conv2DExt2(nn.Module):
#     def __init__(self,*args,**kwargs):
#         super().__init__(self)
#         self.conv2d = nn.ConvTranspose2d(*args,**kwargs)
#     def forward(self, input, output_size=None):
#         if input.ndimensions() == 5:
#             B, F, C, H, W = input.shape
#             input = input.view(-1, C, H, W)
#         return self.conv2d(input, output_size=output_size)

class RapidENet(nn.Module):
    def __init__(self, number_of_classes = 2, input_size=(1, 22, 20), dropout_rate = 0.5):
        
        
        
        super(RapidENet, self).__init__()
        self.number_of_classes = number_of_classes
        self.dropout_rate = dropout_rate

        #self.batchNormScatter2 = nn.BatchNorm2d(1)
        self.scatterConv1 = nn.Sequential(
            nn.ReplicationPad2d(2),
            nn.Conv2d(1, 10, 5), nn.Dropout2d(dropout_rate), nn.MaxPool2d(2), nn.ReLU()
        )
        
        self.batchNormScatter = nn.BatchNorm2d(10)
        
        self.scatterConv2 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(10, 20, 3), nn.Dropout2d(dropout_rate), nn.MaxPool2d(2), nn.ReLU()
        )
        
        
        self.spectrumnConv1 = nn.Sequential(
            nn.ReplicationPad2d(2),
            nn.Conv2d(1, 50, 5), nn.Dropout2d(dropout_rate), nn.MaxPool2d(2), nn.ReLU()
        )
        
        self.batchNormSpectrum = nn.BatchNorm2d(50)
        
        self.spectrumnConv2 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(50, 100, 3), nn.Dropout2d(dropout_rate), nn.MaxPool2d(2), nn.ReLU()
        )
        
        
        
        
        self.lifetimeConv1 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(1, 70, (1, 7)), nn.Dropout2d(dropout_rate), nn.MaxPool2d(2), nn.ReLU()
        )
        
        self.batchNormLifetime1 = nn.BatchNorm2d(70)
        
        self.lifetimeConv2 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(70, 140, (1, 5)), nn.Dropout2d(dropout_rate), nn.MaxPool2d(2), nn.ReLU()
        )
         
        self.batchNormLifetime2 = nn.BatchNorm2d(140)
         
         
        self.lifetimeConv3 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(140, 200, 3), nn.Dropout2d(dropout_rate), nn.MaxPool2d(2), nn.ReLU()
        )
        
        
        # FC layers
        
        self.batchNormFCScatter = nn.BatchNorm1d(3000)
        self.batchNormFCSpectrum = nn.BatchNorm1d(800)
        self.batchNormFCLifetime = nn.BatchNorm1d(400)
         
        self.FCScatter = nn.Sequential(
            nn.Linear(3000, 50), nn.ReLU(), nn.Dropout2d(dropout_rate),
        )
        self.FCSpectrum = nn.Sequential(
            nn.Linear(800, 50), nn.ReLU(), nn.Dropout2d(dropout_rate),
        )
        self.FCLifetime1 = nn.Sequential(
            nn.Linear(400, 50), nn.ReLU(), nn.Dropout2d(dropout_rate),
    
        )
        
        self.FCLifetime2 = nn.Sequential(
            nn.ReLU(), nn.Dropout2d(dropout_rate)
        )
        self.FCSize = nn.Sequential(
            nn.ReLU(), nn.Dropout2d(dropout_rate)
        )
        
        self.batchNormFinal = nn.BatchNorm1d(155)
        
        self.FCFinal = nn.Sequential(
            nn.Linear(155, number_of_classes), nn.ReLU(),
            nn.Softmax(dim = 1)
            
        )
        #self.collection42 = nn.Threshold(0.9, 0)
        
        #self.collection43 = []
        #for i in range(number_of_classes):
        #    self.collection43.append(
        #        nn.Sequential(
        #    nn.Linear(1, 50), nn.ReLU(), nn.Linear(50, 1), nn.ReLU()))
        
        
 

    def forward(self, data_batch):  # red: spec, scat, life1, life2, size
        
        numpart_per_hour = list(map(lambda x: x[0].shape[0], data_batch))
        #print(numpart_per_hour)
        scatters = list(map(lambda x: x[0], data_batch))
        spectrums = list(map(lambda x: x[1], data_batch))
        lifetimes1 = list(map(lambda x: x[2], data_batch))
        lifetimes2 = list(map(lambda x: x[3], data_batch))
        sizes = list(map(lambda x: x[4], data_batch))
        

    
        #y = torch.zeros((len(scatters), self.number_of_classes))   # create a new variable which will contain outputs for each hour
        # x is a data for ONE HOUR
        scatters = [self.scatterConv1(x) for x in scatters] 
        #print(scatters[0].shape)
        scatters = torch.split(self.batchNormScatter(torch.cat(scatters, dim=0)), numpart_per_hour, dim=0)
        scatters = [self.scatterConv2(x) for x in scatters]
        #print(scatters[0].shape)
        
        spectrums = [self.spectrumnConv1(x) for x in spectrums]
        spectrums = torch.split(self.batchNormSpectrum(torch.cat(spectrums, dim=0)), numpart_per_hour, dim=0)
        spectrums = [self.spectrumnConv2(x) for x in spectrums]
        #print(spectrums[0].shape)
        
    
        lifetimes1 = [self.lifetimeConv1(x) for x in lifetimes1]
        lifetimes1 = torch.split(self.batchNormLifetime1(torch.cat(lifetimes1, dim=0)), numpart_per_hour, dim=0)
        lifetimes1 = [self.lifetimeConv2(x) for x in lifetimes1]
        lifetimes1 = torch.split(self.batchNormLifetime2(torch.cat(lifetimes1, dim=0)), numpart_per_hour, dim=0)
        lifetimes1 = [self.lifetimeConv3(x) for x in lifetimes1]
        #print(lifetimes1[0].shape)
        
        
        # PREPARE FOR FC LAYERS
        
        scatters = [x.view(x.shape[0],3000) for x in scatters]
        scatters = torch.split(self.batchNormFCScatter(torch.cat(scatters, dim=0)), numpart_per_hour, dim=0)
        spectrums = [x.view(x.shape[0],800) for x in spectrums]
        spectrums = torch.split(self.batchNormFCSpectrum(torch.cat(spectrums, dim=0)), numpart_per_hour, dim=0)
        lifetimes1 = [x.view(x.shape[0],400) for x in lifetimes1]
        lifetimes1 = torch.split(self.batchNormFCLifetime(torch.cat(lifetimes1, dim=0)), numpart_per_hour, dim=0)
        
        
        scatters = [self.FCScatter(x) for x in scatters]
        spectrums = [self.FCSpectrum(x) for x in spectrums]
        lifetimes1 = [self.FCLifetime1(x) for x in lifetimes1]
        lifetimes2 = [self.FCLifetime2(x) for x in lifetimes2]
        sizes = [self.FCSize(x) for x in sizes]

        features = [ torch.cat((sc, sp, lf1, lf2, sz), dim=1) for (sc, sp, lf1, lf2, sz) in zip(scatters,spectrums,lifetimes1, lifetimes2, sizes)]
        features = torch.split(self.batchNormFinal(torch.cat(features, dim=0)), numpart_per_hour, dim=0)
        ouputs = [self.FCFinal(x) for x in features]
        
        outputs = torch.stack([torch.sum(x,dim=0) for x in ouputs], dim=0)
        outputs = (outputs - torch.mean(outputs, dim=0)) / torch.std(outputs, dim = 0)
        return outputs[:,-1]
        

        


class RapidENetCUDA(nn.Module):
    def __init__(self, obj_loss, number_of_classes = 2, dropout_rate = 0.5, add_loss = None):
        
        
        
        super(RapidENetCUDA, self).__init__()
        self.number_of_classes = number_of_classes
        self.dropout_rate = dropout_rate
        self.obj_loss = obj_loss

        #self.batchNormScatter2 = nn.BatchNorm2d(1)
        self.scatterConv1 = nn.Sequential(
            nn.ReplicationPad2d(2),
            nn.Conv2d(1, 10, 5), nn.Dropout2d(dropout_rate), nn.MaxPool2d(2), nn.ReLU()
        )
        
        self.batchNormScatter = nn.BatchNorm2d(10)
        
        self.scatterConv2 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(10, 20, 3), nn.Dropout2d(dropout_rate), nn.MaxPool2d(2), nn.ReLU()
        )
        
        
        self.spectrumnConv1 = nn.Sequential(
            nn.ReplicationPad2d(2),
            nn.Conv2d(1, 50, 5), nn.Dropout2d(dropout_rate), nn.MaxPool2d(2), nn.ReLU()
        )
        
        self.batchNormSpectrum = nn.BatchNorm2d(50)
        
        self.spectrumnConv2 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(50, 100, 3), nn.Dropout2d(dropout_rate), nn.MaxPool2d(2), nn.ReLU()
        )
        
        
        
        
        self.lifetimeConv1 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(1, 70, (1, 7)), nn.Dropout2d(dropout_rate), nn.MaxPool2d(2), nn.ReLU()
        )
        
        self.batchNormLifetime1 = nn.BatchNorm2d(70)
        
        self.lifetimeConv2 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(70, 140, (1, 5)), nn.Dropout2d(dropout_rate), nn.MaxPool2d(2), nn.ReLU()
        )
         
        self.batchNormLifetime2 = nn.BatchNorm2d(140)
         
         
        self.lifetimeConv3 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(140, 200, 3), nn.Dropout2d(dropout_rate), nn.MaxPool2d(2), nn.ReLU()
        )
        
        
        # FC layers
        
        self.batchNormFCScatter = nn.BatchNorm1d(3000)
        self.batchNormFCSpectrum = nn.BatchNorm1d(800)
        self.batchNormFCLifetime = nn.BatchNorm1d(400)
         
        self.FCScatter = nn.Sequential(
            nn.Linear(3000, 50), nn.ReLU(), nn.Dropout2d(dropout_rate),
        )
        self.FCSpectrum = nn.Sequential(
            nn.Linear(800, 50), nn.ReLU(), nn.Dropout2d(dropout_rate),
        )
        self.FCLifetime1 = nn.Sequential(
            nn.Linear(400, 50), nn.ReLU(), nn.Dropout2d(dropout_rate),
    
        )
        
        self.FCLifetime2 = nn.Sequential(
            nn.ReLU(), nn.Dropout2d(dropout_rate)
        )
        self.FCSize = nn.Sequential(
            nn.ReLU(), nn.Dropout2d(dropout_rate)
        )
        
        self.batchNormFinal = nn.BatchNorm1d(155)
        
        self.FCFinal = nn.Sequential(
            nn.Linear(155, number_of_classes), nn.ReLU(),
            nn.Softmax(dim = 1)
            
        )
        #self.collection42 = nn.Threshold(0.9, 0)
        
        #self.collection43 = []
        #for i in range(number_of_classes):
        #    self.collection43.append(
        #        nn.Sequential(
        #    nn.Linear(1, 50), nn.ReLU(), nn.Linear(50, 1), nn.ReLU()))
        
    
 

    def forward(self, scatters, spectrums, lifetimes1, lifetimes2, sizes, target, weights):  # red: spec, scat, life1, life2, size
        
        
        
        
        #print(len(scatters))
        #print(train_batch_target)
        
        numpart_per_hour = list(map(lambda x: x.shape[0], scatters))
        #y = torch.zeros((len(scatters), self.number_of_classes))   # create a new variable which will contain outputs for each hour
        # x is a data for ONE HOUR
        scatters = [self.scatterConv1(x) for x in scatters]

        #print(sum(numpart_per_hour))
        #catted = self.batchNormScatter(torch.cat(scatters, dim=0))
        #print(catted.shape)
        scatters = torch.split(self.batchNormScatter(torch.cat(scatters, dim=0)), numpart_per_hour, dim=0)
        scatters = [self.scatterConv2(x) for x in scatters]
        #print(scatters[0].shape)
        #(spectrums)
        spectrums = [self.spectrumnConv1(x) for x in spectrums]
        spectrums = torch.split(self.batchNormSpectrum(torch.cat(spectrums, dim=0)), numpart_per_hour, dim=0)
        spectrums = [self.spectrumnConv2(x) for x in spectrums]
        #print(spectrums[0].shape)
        
    
        lifetimes1 = [self.lifetimeConv1(x) for x in lifetimes1]
        lifetimes1 = torch.split(self.batchNormLifetime1(torch.cat(lifetimes1, dim=0)), numpart_per_hour, dim=0)
        lifetimes1 = [self.lifetimeConv2(x) for x in lifetimes1]
        lifetimes1 = torch.split(self.batchNormLifetime2(torch.cat(lifetimes1, dim=0)), numpart_per_hour, dim=0)
        lifetimes1 = [self.lifetimeConv3(x) for x in lifetimes1]
        #print(lifetimes1[0].shape)
        
        
        # PREPARE FOR FC LAYERS
        
        scatters = [x.view(x.shape[0],3000) for x in scatters]
        scatters = torch.split(self.batchNormFCScatter(torch.cat(scatters, dim=0)), numpart_per_hour, dim=0)
        spectrums = [x.view(x.shape[0],800) for x in spectrums]
        spectrums = torch.split(self.batchNormFCSpectrum(torch.cat(spectrums, dim=0)), numpart_per_hour, dim=0)
        lifetimes1 = [x.view(x.shape[0],400) for x in lifetimes1]
        lifetimes1 = torch.split(self.batchNormFCLifetime(torch.cat(lifetimes1, dim=0)), numpart_per_hour, dim=0)
        
        
        scatters = [self.FCScatter(x) for x in scatters]
        spectrums = [self.FCSpectrum(x) for x in spectrums]
        lifetimes1 = [self.FCLifetime1(x) for x in lifetimes1]
        lifetimes2 = [self.FCLifetime2(x) for x in lifetimes2]
        sizes = [self.FCSize(x) for x in sizes]


        features = [ torch.cat((sc, sp, lf1, lf2, sz), dim=1) for (sc, sp, lf1, lf2, sz) in zip(scatters,spectrums,lifetimes1, lifetimes2, sizes)]
        features = torch.split(self.batchNormFinal(torch.cat(features, dim=0)), numpart_per_hour, dim=0)
        ouputs = [self.FCFinal(x) for x in features]
        
        outputs = torch.stack([torch.sum(x,dim=0) for x in ouputs], dim=0)
        outputs = (outputs - torch.mean(outputs, dim=0)) / torch.std(outputs, dim = 0)

        
        return outputs[:,-1]



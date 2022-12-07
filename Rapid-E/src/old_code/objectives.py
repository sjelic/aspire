# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 00:44:30 2020

@author: sjelic
"""

import torch




# class  WeightedSELoss():
    
#     def __init__(self, name = 'WeightedSELoss', sense= 'min', reduction='sum', selection = False):
#         self.reduction = reduction
#         self.sense = sense
#         self.name = name
#         self.selection = selection

    
#     def __call__(self,output, target, weights):
#         #print(weights.type())
#         #print(target.type())
#         #print(output.type())
#         print(weights.shape)
#         print(target.shape)
#         print(output.shape)
#         # print(((output - target)**2).type())
#         return torch.dot(weights,(output - target)**2)
    
    


class WeightedSELoss(torch.nn.Module):

    def __init__(self, name = 'WeightedSELoss', sense= 'min', reduction='sum', selection = False):
        super(WeightedSELoss, self).__init__()
        self.reduction = reduction
        self.sense = sense
        self.name = name
        self.selection = selection
    #def forward(self, output, target, weights):
    def forward(self, target,weights):
        #target = torch.cat(target)
        #weights = torch.cat(weights)
        #print(target)
        #print(weights)
        target = torch.Tensor(list(map(lambda x: x[0], target)))
        #print(target)
        # print('Calc:')
        # print('output:')
        # print(output)
        # print('target:')
        # print(target)
        # print('weights:')
        # print(weights)
        #return torch.tensor([0])
        #res = torch.dot(weights,(target - target)**2).reshape(1)
        #print(res.shape)
        return torch.Tensor([0])
        #return torch.dot(weights,(output - target)**2)




class PearsonCorrelationLoss(torch.nn.Module):

    def __init__(self, name='PearsonCorrelationLoss', sense= 'max', reduction='mean', selection = False):
        super(PearsonCorrelationLoss, self).__init__()
        self.reduction = reduction
        self.sense = sense
        self.name = name
        self.selection = selection
        
    def forward(self, x, y, weights = None):                                 
        x = x - torch.mean(x)
        y = y - torch.mean(y)
        return torch.sum(x * y) / (torch.sqrt(torch.sum(x ** 2)) * torch.sqrt(torch.sum(y ** 2)))

# class PearsonCorrelationLoss():
#     def __init__(self, name='PearsonCorrelationLoss', sense= 'max', reduction='mean', selection = False):
#         self.reduction = reduction
#         self.sense = sense
#         self.name = name
#         self.selection = selection
    
#     def __call__(self, x, y, weights = None):
#         x = x - torch.mean(x)
#         y = y - torch.mean(y)
#         return torch.sum(x * y) / (torch.sqrt(torch.sum(x ** 2)) * torch.sqrt(torch.sum(y ** 2)))
    

# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 16:35:22 2020

@author: sjelic
"""
import os
import pandas as pd
#import numpy as np
#from datetime import datetime, timedelta
import pickle
import torch
from torch import nn
#import matplotlib.pyplot as plt
#import functools as fnc
from type_runner import split_train_test

GPU = 1

class Net(nn.Module):
    def __init__(self, number_of_classes = 2, input_size=(1, 22, 20)):
        super(Net, self).__init__()
        self.collection11 = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.ReplicationPad2d(2),
            nn.Conv2d(input_size[0], 10, 5), nn.Dropout2d(), nn.MaxPool2d(2), nn.ReLU(),

            nn.BatchNorm2d(10),
            nn.ReplicationPad2d(1),
            nn.Conv2d(10, 20, 3), nn.Dropout2d(), nn.MaxPool2d(2), nn.ReLU()
        )
        self.collection12 = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.ReplicationPad2d(2),
            nn.Conv2d(input_size[0], 50, 5), nn.Dropout2d(), nn.MaxPool2d(2), nn.ReLU(),

            nn.BatchNorm2d(50),
            nn.ReplicationPad2d(1),
            nn.Conv2d(50, 100, 3), nn.Dropout2d(), nn.MaxPool2d(2), nn.ReLU()
        )
        self.collection13 = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.ReplicationPad2d(1),
            nn.Conv2d(1, 70, (1, 7)), nn.Dropout2d(), nn.MaxPool2d(2), nn.ReLU(),

            nn.BatchNorm2d(70),
            nn.ReplicationPad2d(1),
            nn.Conv2d(70, 140, (1, 5)), nn.Dropout2d(), nn.MaxPool2d(2), nn.ReLU(),

            nn.BatchNorm2d(140),
            nn.ReplicationPad2d(1),
            nn.Conv2d(140, 200, 3), nn.Dropout2d(), nn.MaxPool2d(2), nn.ReLU()
        )
        self.collection21 = nn.Sequential(
            nn.Linear(3000, 50), nn.ReLU(), nn.Dropout2d(),
        )
        self.collection22 = nn.Sequential(
            nn.Linear(800, 50), nn.ReLU(), nn.Dropout2d(),
        )
        self.collection23 = nn.Sequential(
            nn.Linear(400, 50), nn.ReLU(), nn.Dropout2d(),
        )
        self.collection31 = nn.Sequential(
            nn.BatchNorm1d(4), nn.ReLU(), nn.Dropout2d(),
        )
        self.collection32 = nn.Sequential(
            nn.BatchNorm1d(1), nn.ReLU(), nn.Dropout2d(),
        )
        self.collection41 = nn.Sequential(
            nn.Linear(3 * 50 + 4 + 1, number_of_classes), nn.ReLU(),
            nn.Softmax(dim = 1),
            
        )
        self.collection42 = nn.Threshold(0.9, 0)
        
        self.collection43 = []
        for i in range(number_of_classes):
            self.collection43.append(
                nn.Sequential(
            nn.Linear(1, 50), nn.ReLU(), nn.Linear(50, 1), nn.ReLU()))
        
        
        
    #def init_weights(self):
    #    for ly in self.collection42:
    #        ly.weight.data.fill_(3)
            
        

    def forward(self, x1_o, x2_o, x3_o, x4_o, x5_o, var):  # red: spec, scat, life1, life2, size
        y = torch.zeros((len(x1_o), number_of_classes))   # create a new variable which will contain outputs for each hour

        for i in range(len(x1_o)):          # go through each batch
            x1 = x1_o[i]
            x2 = x2_o[i]
            x3 = x3_o[i]
            x4 = x4_o[i]
            x5 = x5_o[i]

            # do convolutions on scatter lifetime and spectrum data
            x2 = self.collection11(x2)
            x1 = self.collection12(x1)
            x3 = self.collection13(x3)

            # transform feature maps into vectors for FC layers
            x2 = x2.view(-1, 3000)
            x1 = x1.view(-1, 800)
            x3 = x3.view(-1, 400)

            # FC layers for each of the inputs
            x2 = self.collection21(x2)
            x1 = self.collection22(x1)
            x3 = self.collection23(x3)
            x4 = self.collection31(x4)
            x5 = self.collection32(x5)

            # concatenate the features
            x = torch.cat((x1, x2, x3, x4, x5), 1)

            # classify with the softmax function
            x = self.collection41(x)
            #print(x.shape)
            #x = self.collection42(x)

            # sum the results of one hour and add it to the new variable y
            y[i,:] = torch.sum(x, 0)
            #print(x)
            #y[i,:] = x
           

        # if hirst data for the batch size has a nonzero element, normalize the output of the model
        # if var == 1:
        #     y_min = y[:, 0] - torch.min(y[:, 0])
        #     y_norm = y_min / torch.max(y_min)
        # else:       # else just do softmax on the data to pin the output for each hour into 0-1 and sum(output(hour)) = 1 (e.g. AMB = 0.3, OTH = 0.7) and minimize the value for AMB
        #     y_soft = self.collection42(y)
        #     y_norm = y_soft[:, 0]
        y = self.collection42(y)
        for i in range(len(x1_o)):
            for j in range(number_of_classes):
                y[i,j] = self.collection43[j](torch.tensor([y[i,j]], dtype=torch.float32))
                #print(y[i,j])
        #print(y.shape)
        #print(y[:,0])
        
        y = (y - torch.mean(y))/torch.std(y)
        return y[:,0]
    
    
    
loss_fn = torch.nn.MSELoss()    
    
    

#hirst = pd.read_excel("./Libraries/HIRST_2019021601_2019101706.xlsx")
calibs = ["2019-02-18 16", "2019-02-25 12", "2019-03-11 12", "2019-03-15 12", "2019-03-15 13", 
          "2019-03-19 07", "2019-03-19 08", "2019-03-25 08","2019-03-25 09", "2019-03-26 08", 
          "2019-03-29 08", "2019-04-04 10", "2019-04-04 11", "2019-04-09 07", "2019-04-15 07", 
          "2019-04-18 16" , "2019-04-22 08", "2019-04-22 14", "2019-04-29 10", "2019-05-03 08", 
          "2019-05-07 16", "2019-05-10 16", "2019-05-24 13", "2019-06-03 11", "2019-06-13 15", 
          '2019-05-15 03', '2019-05-15 04', '2019-08-03 03', '2019-09-23 04', '2019-09-24 21', 
          '2019-09-27 13', '2019-10-03 02', '2019-10-03 06', '2019-10-06 00', '2019-10-07 06', 
          '2019-10-11 07', '2019-10-11 08', '2019-10-12 02', '2019-10-13 03']

#define parameters for training
number_of_classes = 2
model_name = 'Ambrosia_vs_all'
pollen_type = 'AMBR'

#start_date = '201908050000'
#end_date = '201910050559'

# get a list of data files
#hours = os.listdir('../data/novi_sad_2019_')
#hours = sorted(hours)

model = Net(number_of_classes)

model.load_state_dict(torch.load('./models/novi_sad/model_pollen_types_ver0/' + model_name + '.pth', map_location=lambda storage, loc: storage), strict=False)

#model.init_weights()
#model.cuda(GPU)

#optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)


#hirst['Unnamed: 0'] = list(map(lambda x: datetime.strptime(x[:-6],'%Y-%m-%d %H:%M:%S'),list(hirst['Unnamed: 0'])))
#y = hirst.loc[(hirst['Unnamed: 0'] >= datetime(2019,8,13,0,0,0)) & (hirst['Unnamed: 0'] <= datetime(2019,8,13,23,0,0))]['AMBR'].astype(float)

# def param_shapes(model):
#     p_shapes = []
#     for x in model.parameters():
#         p_shapes.append(x.detach().numpy().shape)
#     return p_shapes

dir_path = '../data/novi_sad_selected_day/'
HIRST_DATA_PATH = "../Libraries/HIRST_2019021601_2019101706.xlsx"

#files = os.listdir(dir_path)



def load_datasets(dir_path, hirst_data_path):
    spectrum_tensor_train, scatter_tensor_train, lifetime_tensor1_train, lifetime_tensor2_train, size_tensor_train = [], [], [], [], []
    spectrum_tensor_valid, scatter_tensor_valid, lifetime_tensor1_valid, lifetime_tensor2_valid, size_tensor_valid = [], [], [], [], []

    
    train_set, valid_set = split_train_test(dir_path, hirst_data_path)
    #print(len(valid_set))
    #train_set = list(map(lambda x: x + '.pkl', train_set))
    #valid_set = list(map(lambda x: x + '.pkl', valid_set))
    
    hirst = pd.read_excel(hirst_data_path)
    #print(hirst)
    hirst['Unnamed: 0'] = list(map(lambda x: x[:-12],list(hirst['Unnamed: 0'])))
    #hirst['Unnamed: 0'] = hirst['Unnamed: 0'].astype(string)

    #print(hirst)
    y_norm_train = torch.tensor(list(hirst[hirst["Unnamed: 0"].isin(train_set)]['AMBR']), dtype=torch.float64)
    #print(y_norm_train)
    #print(hirst["Unnamed: 0"].isin(train_set))
    y_norm_train_mean = torch.mean(y_norm_train)
    y_norm_train_std = torch.std(y_norm_train)
    #y_norm_valid_mean = torch.mean(y_norm_valid)
    #y_norm_valid_std = torch.std(y_norm_valid)
    
    y_norm_train = (y_norm_train - y_norm_train_mean)/y_norm_train_std
    
    y_norm_valid = torch.tensor(list(hirst[hirst["Unnamed: 0"].isin(valid_set)]['AMBR']), dtype=torch.float64)
    y_norm_valid_mean = torch.mean(y_norm_valid)
    y_norm_valid_std = torch.std(y_norm_valid)
    y_norm_valid = (y_norm_valid - y_norm_valid_mean)/y_norm_valid_std
    #print(y_norm_valid.shape)
    train_set = list(map(lambda x: x + '.pkl', train_set))
    valid_set = list(map(lambda x: x + '.pkl', valid_set))

    for fn in train_set:
        with open(os.path.join(dir_path,fn), 'rb') as fp:
            data = pickle.load(fp)
            spectrum_tensor_train.append(torch.Tensor(data[1]).unsqueeze_(0).permute(1, 0, 2, 3))
            scatter_tensor_train.append(torch.Tensor(data[0]).unsqueeze_(0).permute(1, 0, 2, 3))
            lifetime_tensor1_train.append(torch.Tensor(data[2]).unsqueeze_(0).permute(1, 0, 2, 3))
            lifetime_tensor2_train.append(torch.Tensor(data[3]))
            size_tensor_train.append(torch.Tensor(data[4]).unsqueeze_(0).permute(1, 0))
    
    for fn in valid_set:
        with open(os.path.join(dir_path,fn), 'rb') as fp:
            data = pickle.load(fp)
            spectrum_tensor_valid.append(torch.Tensor(data[1]).unsqueeze_(0).permute(1, 0, 2, 3))
            scatter_tensor_valid.append(torch.Tensor(data[0]).unsqueeze_(0).permute(1, 0, 2, 3))
            lifetime_tensor1_valid.append(torch.Tensor(data[2]).unsqueeze_(0).permute(1, 0, 2, 3))
            lifetime_tensor2_valid.append(torch.Tensor(data[3]))
            size_tensor_valid.append(torch.Tensor(data[4]).unsqueeze_(0).permute(1, 0))
    
    
    return spectrum_tensor_train, scatter_tensor_train, lifetime_tensor1_train, lifetime_tensor2_train, size_tensor_train, spectrum_tensor_valid, scatter_tensor_valid, lifetime_tensor1_valid, lifetime_tensor2_valid, size_tensor_valid, y_norm_train, y_norm_valid

#def vectorize_params(model_par, shap)
#for 



# def closure():
      
#     output_train = model(spectrum_tensor_train, scatter_train, lifetime_tensor1_train, lifetime_tensor2_train, size_tensor_train, 1).double()
#     loss = loss_fn(output, y_norm)
#     l1_par = 0
#     l2_par = 0
#     l1_penal = 0;
#     l2_penal = 0;
#     for p in model.parameters():
#         l1_penal += l1_par * torch.sum(torch.abs(p))
#         l2_penal += l2_par * torch.sum(p.data ** 2)
    
#     loss += l1_penal + l2_penal
  
    
    
#     loss.backward()
#     return loss, corr

def regularized_loss(output, y, l1_par, l2_par):
    loss = loss_fn(output, y)
    l1_penal = 0.0;
    l2_penal = 0.0;
    for p in model.parameters():
        l1_penal += l1_par * torch.sum(torch.abs(p))
        #print(l1_penal.item())
        l2_penal += l2_par * torch.sum(p.data ** 2)
        #print(l2_penal.item())
    loss += l1_penal.item() + l2_penal.item()
    return loss

def correlation(output, y):
    vx = output - torch.mean(output)
    vy = y - torch.mean(y)

    corr = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
    
    return corr
    
    

def adam_optimizer(model, spectrum_tensor_train, scatter_tensor_train, lifetime_tensor1_train, lifetime_tensor2_train, size_tensor_train, spectrum_tensor_valid, scatter_tensor_valid, lifetime_tensor1_valid, lifetime_tensor2_valid, size_tensor_valid, y_norm_train, y_norm_valid, maxiter, tol):
    i=0
    #learning_rate = 1e-2
    #prev_value = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=4, factor=0.5, verbose=True)
    prev_loss = 0;
    print(sum([x.shape[0] for x in spectrum_tensor_train]))
    
    
    while(i <= maxiter):
        
        optimizer.zero_grad()
        output_train = model(spectrum_tensor_train, scatter_tensor_train, lifetime_tensor1_train, lifetime_tensor2_train, size_tensor_train, 1).double()
        output_valid = model(spectrum_tensor_valid, scatter_tensor_valid, lifetime_tensor1_valid, lifetime_tensor2_valid, size_tensor_valid, 1).double()
        print(output_train.shape)
        print(output_valid.shape)
        loss_train = regularized_loss(output_train, y_norm_train,0,0)
        loss_valid = regularized_loss(output_valid, y_norm_valid,0,0)
        
        corr_train = correlation(output_train, y_norm_train)
        corr_valid = correlation(output_valid, y_norm_valid)        
        
        loss_train.backward()
        
        
        
        if (abs(prev_loss - loss_train) <= tol):
            print("STOP. No improvements!")
            break
        print("Epoch " + str(i+1))
        print("\tTrain loss: " + str(loss_train.item()))
        print("\tValidation loss: " + str(loss_valid.item()))
        print("\tTrain correlation: " + str(corr_train.item()))
        print("\tValidation correlation: " + str(corr_valid.item()))
        #print("Loss: " + str(loss))
        #optimizer.zero_grad()

        #loss.backward()
        optimizer.step(lambda: loss_train)
        scheduler.step(loss_train)
        prev_loss = loss_train

#mulsti_start_gradient_descent(model,10)
    
def train_model(model,spectrum_tensor_train, scatter_tensor_train, lifetime_tensor1_train, lifetime_tensor2_train, size_tensor_train, spectrum_tensor_valid, scatter_tensor_valid, lifetime_tensor1_valid, lifetime_tensor2_valid, size_tensor_valid, y_norm_train, y_norm_valid):
    
    MAX_ITER = 100
    adam_optimizer(model, spectrum_tensor_train, scatter_tensor_train, lifetime_tensor1_train, lifetime_tensor2_train, size_tensor_train, spectrum_tensor_valid, scatter_tensor_valid, lifetime_tensor1_valid, lifetime_tensor2_valid, size_tensor_valid, y_norm_train, y_norm_valid, MAX_ITER, 1e-4)


#yten = torch.tensor(np.array(y))
#y_norm = yten
#y_norm = (yten - torch.mean(yten))/torch.std(yten)

spectrum_tensor_train, scatter_tensor_train, lifetime_tensor1_train, lifetime_tensor2_train, size_tensor_train, spectrum_tensor_valid, scatter_tensor_valid, lifetime_tensor1_valid, lifetime_tensor2_valid, size_tensor_valid, y_norm_train, y_norm_valid = load_datasets(
    '../../data/novi_sad_2019_/', "./Libraries/HIRST_2019021601_2019101706.xlsx")


train_model(model, spectrum_tensor_train, scatter_tensor_train, lifetime_tensor1_train, lifetime_tensor2_train, size_tensor_train, spectrum_tensor_valid, scatter_tensor_valid, lifetime_tensor1_valid, lifetime_tensor2_valid, size_tensor_valid, y_norm_train, y_norm_valid)
#num_of_iter = 50

# def set_params(model, params):
#     for i, p in enumerate(model.parameters()):
#         p.data = params[i]

# def set_params(copy, params):
#     for i, p in enumerate(model.parameters()):
#         p.data = params[i]

# def get_params(model):
#     params = []
#     for p in model.parameters():
#         pp = p.data.clone().detach().requires_grad_(True)
#         params.append(pp)
#     return params


# #param = list(model.parameters())


# def gradient_step(model, lr, grad_norm):
#     for p in model.parameters():
#         p -= (lr/grad_norm) * p.grad

# def gradient_norm(model):
#     nrm = 0
#     for p in model.parameters():
#         nrm += p.grad.data.norm(2).item() ** 2
#     return np.sqrt(nrm)

# def zero_gradients(model):
#     for p in model.parameters():
#         p.grad.data.zero_()


# def armijo_goldstein_lr(model, loss_fn, curr_loss, grad_norm, y_norm):
    
#     lr = 1;
#     p_current = get_params(model)
#     c = 0.0001
#     t = c * (grad_norm ** 2)
#     count = 1
#     while(True):
#         if (count > 5):
#             c = c * 0.5
#            # print(c)
#             t = c * (grad_norm ** 2)
#             count = 1
#         if (c <= 1e-6):
#             return 0
#         gradient_step(model, lr, grad_norm)
#         new_out = model(spectrum_tensor, scatter_tensor, lifetime_tensor1, lifetime_tensor2, size_tensor, 1).double()
#         new_loss = loss_fn(new_out,y_norm)
#         #print("Current loss: " + str(curr_loss))
#         #print("New loss: " + str(new_loss))
#         if (curr_loss - new_loss < lr * t):
#             lr = 0.5 * lr
#             #print(lr)
#             set_params(model, p_current)
#             count+=1
#         else:
#             set_params(model, p_current)
#             return lr
            
        
        
    
# def simple_line_search(model, loss_fn, curr_loss, grad_norm, y_norm):
#     lr = 1;
#     #p_current = get_params(model)
   
#     count = 1
#     while(True):
#         if (count > 10):
#             return curr_loss;
          
#         gradient_step(model, lr, grad_norm)
#         new_out = model(spectrum_tensor, scatter_tensor, lifetime_tensor1, lifetime_tensor2, size_tensor, 1).double()
#         new_loss = loss_fn(new_out,y_norm)
#         #print("Current loss: " + str(curr_loss))
#         #print("New loss: " + str(new_loss))
#         if (curr_loss <= new_loss):
#             lr = 0.5 * lr
#             #print(lr)
#             #set_params(model, p_current)
#             count+=1
#         else:
#             #set_params(model, p_current)
#             return new_loss



# #for i in range(num_of_iter):
# #    y_pred = model(spectrum_tensor,scatter_tensor, lifetime_tensor1,lifetime_tensor2, size_tensor,1)

# #model.train()
# #model_parameters = filter(lambda p: p.requires_grad, model.parameters())
# #n_params = sum([np.prod(p.size()) for p in model_parameters])




# def gradient_descent(model, maxiter):
#     i=0
#     #learning_rate = 1e-2
#     #prev_value = 0
#     while(i <= maxiter):
        
#         output = model(spectrum_tensor, scatter_tensor, lifetime_tensor1, lifetime_tensor2, size_tensor, 1).double()
        
        
        
#         loss = loss_fn(output, y_norm)
        
#         #if (abs(loss - prev_value) <= 1e-15) :
#         #    print("Loss value not changes. EXIT")
#         #    break
        
#        # prev_value = loss
        
#         print("Iteration " + str(i+1) + ": " + str(loss))
#         #optimizer.zero_grad()
        
#         # racunanje gradijenta
#         loss.backward()
        
#         with torch.no_grad():
            
#             grd_norm = gradient_norm(model)
#             print("Gradient norm: " + str(grd_norm))
#             if (grd_norm <= 1e-5):
#                 print("FINISHED: Gradient zero")
#             new_loss = simple_line_search(model, loss_fn, loss, grd_norm, y_norm)
#             zero_gradients(model)
#             if (abs(loss - new_loss) <= 1e-12):
#                 print("FINISED")
#                 break;
#             #gradient_step(model, lr, grd_norm)
            
#             i+=1
#     return new_loss

    
# def randomly_reinitialize(params):
#     pps = []
#     for p in params:
#         #print(p.shape)
#         pp = p.data.clone().detach()
#         pp += torch.FloatTensor(p.data.shape).uniform_(1e-7,1e-4)
#         pps.append(pp.requires_grad_(True))
#     return pps

# def mulsti_start_gradient_descent(model, max_round):
#     #initial
#     best_loss = gradient_descent(model,50)
#     best_param = get_params(model);
    
    
#     i = 1
#     while(i <= max_round):
#         print("Start round " + str(i))
#         params = randomly_reinitialize(best_param)
#         set_params(model, params)
#         #print(params)
#         loss = gradient_descent(model,50)
#         if (loss < best_loss):
#             best_loss = loss
#             best_param = model.parameters()
#             i=1
#             print("RESTART WITH NEW BEST")
#             continue;
#         i+=1
        
#         #randomly_reinitialize(model)
#         #gradient_descent(model)
        


# # ss = list(model.parameters())
# # p_shapes = param_shapes(model)

# # params = ss[0].detach().numpy().reshape(-1)
# # for i in range(1,len(p_shapes)):
# #         params = np.concatenate((params,ss[i].detach().numpy().reshape(-1)))
    

import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import itertools
from sklearn.metrics import confusion_matrix as CM
from sklearn.metrics import accuracy_score as acc
import torch
from torch import nn

def confusion_matrix(save, cm, classes, normalize, klasa, title='Confusion matrix', cmap=plt.cm.Blues):

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=((20, 20)))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=20)
    plt.yticks(tick_marks, classes, fontsize=20)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    
    matplotlib.rcParams.update({'font.size': 20})
    plt.tight_layout()
    plt.ylabel('True label', fontsize=20)
    plt.xlabel('Predicted label', fontsize=20)
    if save:
        plt.savefig('/home/pedjamat/Documents/Faks/Evolucija/models/cm/' + str(klasa) + '.png')
    else:
        plt.show()


# which model to test
pollen_type = 'Urticaceae'

path = '/home/pedjamat/Documents/Pollen/data/'    #path to folder containing data pickle file and another folder with json files
os.chdir(path + 'data ver4 SPt')        #path to .json files
files = sorted(os.listdir())
with open (path + 'data ver4 SPt 5 inputs.pckl', 'rb') as fp:    #path to .pckl file
    dataa = pickle.load(fp)

data = [[[], []], [[], []], [[], []], [[], []], [[], []]]
for i, file in enumerate(files):

    if file.split(".")[0] == pollen_type:
        det = 0
    else:
        det = 1

    data[0][det] += dataa[0][i]
    data[1][det] += dataa[1][i]
    data[2][det] += dataa[2][i]
    data[3][det] += dataa[3][i]
    data[4][det] += dataa[4][i]

    print(len(data[0][det]))
    

#podela na train-test-val

train = [[], [], [], [], []]
test = [[], [], [], [], []]
val = [[], [], [], [], []]

for i in range(len(data[0])):
    k = np.int(np.round(0.9*len(data[0][i])))
    
    for p in range(len(data)):
        train[p] += data[p][i][:k]
        test[p].append(data[p][i][k:])
        val[p] += data[p][i][k:]
        
#labels for validation

l = np.zeros((len(val[0]), 1)).flatten()
b = 0
for i in range(len(test[0])):
    l[b:b+len(test[0][i])] = i
    b += len(test[0][i])
    
#making tensors
train_tensor = [torch.Tensor(train[0]).unsqueeze_(0).permute(1,0,2,3), 
              torch.Tensor(train[1]).unsqueeze_(0).permute(1,0),
              torch.Tensor(train[2]).unsqueeze_(0).permute(1,0,2,3),
              torch.Tensor(train[3]).unsqueeze_(0).permute(1,0,2,3),
              torch.Tensor(train[4])]

val_tensor = [torch.Tensor(val[0]).unsqueeze_(0).permute(1,0,2,3), 
              torch.Tensor(val[1]).unsqueeze_(0).permute(1,0),
              torch.Tensor(val[2]).unsqueeze_(0).permute(1,0,2,3),
              torch.Tensor(val[3]).unsqueeze_(0).permute(1,0,2,3),
              torch.Tensor(val[4])]
y_val_tensor = torch.LongTensor(l)

test_tensor = [[],[],[],[],[]]
for i in range(len(test[0])):
    test_tensor[0].append(torch.Tensor(test[0][i]).unsqueeze_(0).permute(1,0,2,3))
    test_tensor[1].append(torch.Tensor(test[1][i]).unsqueeze_(0).permute(1,0))
    test_tensor[2].append(torch.Tensor(test[2][i]).unsqueeze_(0).permute(1,0,2,3))
    test_tensor[3].append(torch.Tensor(test[3][i]).unsqueeze_(0).permute(1,0,2,3))
    test_tensor[4].append(torch.Tensor(test[4][i]))
    
    
#defining NN architecture
class Net(nn.Module):
    def __init__(self, input_size=(1,22, 20)):
        super(Net, self).__init__()
        self.collection11 = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.ReplicationPad2d(2),
            nn.Conv2d(1, 10, 5), nn.Dropout2d(), nn.MaxPool2d(2), nn.ReLU(),
            
            nn.BatchNorm2d(10),
            nn.ReplicationPad2d(1),
            nn.Conv2d(10,20,3), nn.Dropout2d(), nn.MaxPool2d(2), nn.ReLU()
        )
        
        self.collection12 = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.ReplicationPad2d(2),
            nn.Conv2d(input_size[0], 50, 5), nn.Dropout2d(), nn.MaxPool2d(2), nn.ReLU(),
            
            nn.BatchNorm2d(50),
            nn.ReplicationPad2d(1),
            nn.Conv2d(50,100,3), nn.Dropout2d(), nn.MaxPool2d(2), nn.ReLU()
        )
        
        self.collection13 = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.ReplicationPad2d(1),
            nn.Conv2d(1,70,(1, 7)), nn.Dropout2d(), nn.MaxPool2d(2), nn.ReLU(),
            
            nn.BatchNorm2d(70),
            nn.ReplicationPad2d(1),
            nn.Conv2d(70, 140,(1, 5)), nn.Dropout2d(), nn.MaxPool2d(2), nn.ReLU(), 
            
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
            nn.Linear(3*50+4+1, 2), nn.ReLU(),
            nn.LogSoftmax(dim=1)
        )
    
    def forward(self, x1, x2, x3, x4, x5):  #red: spec, scat, life1, life2, size
        x2 = self.collection11(x2)
        x1 = self.collection12(x1)
        x3 = self.collection13(x3)
        
        x2 = x2.view(-1, 3000)
        x1 = x1.view(-1, 800)
        x3 = x3.view(-1, 400)
        
        x2 = self.collection21(x2)
        x1 = self.collection22(x1)
        x3 = self.collection23(x3)

        x4 = self.collection31(x4)
        x5 = self.collection32(x5)
        
        x = torch.cat((x1, x2, x3, x4, x5), 1)
        
        x = self.collection41(x)
        return x
model = Net()

#load model
model_name = 'r_Urticaceae_season'
#change path to your model
model.load_state_dict(torch.load('/home/pedjamat/Documents/Faks/Evolucija/models/' + model_name + '.pth', map_location=lambda storage, loc: storage))
model.eval()

#get outputs
outputs = model(val_tensor[3], val_tensor[0], val_tensor[2], val_tensor[4], val_tensor[1])
_, predicted = torch.max(outputs.data, 1)

#calculate CM and accur
cm = CM(l, predicted.numpy())
accur = acc(l, predicted.numpy())
title = "Accuracy: " + str('%.2f'%(accur*100))
#in this function change the path where you want to save your CMs
confusion_matrix(save = True, cm = cm, classes = [model_name, "Other"], normalize = True, klasa = model_name, title = title)

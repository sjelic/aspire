import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
import torch
from torch import nn
import matplotlib.pyplot as plt

GPU = 1

# defining the network
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
            nn.LogSoftmax(dim = 1)
        )
        self.collection42 = nn.Softmax(dim = 1)

    def forward(self, x1_o, x2_o, x3_o, x4_o, x5_o, var):  # red: spec, scat, life1, life2, size
        y = torch.zeros((len(x1_o), 2))   # create a new variable which will contain outputs for each hour

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
            x = self.collection42(x)

            # sum the results of one hour and add it to the new variable y
            x = torch.sum(x, 0)
            y[i, :] = x[:2]

        # if hirst data for the batch size has a nonzero element, normalize the output of the model
        if var == 1:
            y_min = y[:, 0] - torch.min(y[:, 0])
            y_norm = y_min / torch.max(y_min)
        else:       # else just do softmax on the data to pin the output for each hour into 0-1 and sum(output(hour)) = 1 (e.g. AMB = 0.3, OTH = 0.7) and minimize the value for AMB
            y_soft = self.collection42(y)
            y_norm = y_soft[:, 0]

        return y_norm
    
hirst = pd.read_excel("../Libraries/HIRST_2019021601_2019101706.xlsx")
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

start_date = '201908050000'
end_date = '201910050559'

# get a list of data files
hours = os.listdir('../data/novi_sad_2019')
hours = sorted(hours)

model = Net(number_of_classes)
model.load_state_dict(torch.load('models/' + model_name + '.pth', map_location=lambda storage, loc: storage))
model.cuda(GPU)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
loss_fn = torch.nn.MSELoss()

batch_size = 24
batch_counter = 0
batch, timestamps, loss_list = [], [], []


try:
    while True:
        for hour in hours:
            dt = hour[:4] + hour[5:7] + hour[8:10] + hour[11:13] + '00'

            if (dt >= start_date) and (dt <= end_date)  and hour.split(".")[0] not in calibs:

                with open('../data/novi_sad_2019/' + hour, 'rb') as fp:
                    data = pickle.load(fp)
                if len(data[0])>1:
                    timestamps.append(hour.split(".")[0] + ':00:00+00:00')
                    batch.append(data)

                # CODE FOR RETRAINING
                if len(batch) == batch_size:

                    batch_counter += 1

                    # GETTING REAL VALUES FROM HIRST
                    timestamp_indices = np.array([np.where(hirst['Unnamed: 0'] == timestamps[i])[0][0] for i in range(batch_size)])
                    y_true = hirst[pollen_type][timestamp_indices].values

                    # PREPROCESS: if there is a nonzero element in real data, normalize into 0-1 range
                    if (y_true != 0).any():
                        var = 1
                        y_true_min = y_true - np.min(y_true)
                        y_true_norm = y_true_min/np.max(y_true)
                        y_true_tensor = torch.Tensor(y_true_norm)
                    else:
                        var = 0
                        y_true_tensor = torch.Tensor(y_true)

                    spectrum_tensor, scatter_tensor, lifetime_tensor1, lifetime_tensor2, size_tensor = [], [], [], [], []
                    for i in range(batch_size):
                        if len(batch[i][1]) != 0:
                            spectrum_tensor.append(torch.Tensor(batch[i][1]).unsqueeze_(0).permute(1, 0, 2, 3).cuda(GPU))
                            scatter_tensor.append(torch.Tensor(batch[i][0]).unsqueeze_(0).permute(1, 0, 2, 3).cuda(GPU))
                            lifetime_tensor1.append(torch.Tensor(batch[i][2]).unsqueeze_(0).permute(1, 0, 2, 3).cuda(GPU))
                            lifetime_tensor2.append(torch.Tensor(batch[i][3]).cuda(GPU))
                            size_tensor.append(torch.Tensor(batch[i][4]).unsqueeze_(0).permute(1, 0).cuda(GPU))

                    model.train()
                    output = model(spectrum_tensor, scatter_tensor, lifetime_tensor1, lifetime_tensor2, size_tensor, var)

                    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
                    loss_fn = torch.nn.MSELoss()

                    loss = loss_fn(output, y_true_tensor)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    loss = loss.detach().numpy()
                    loss_list.append(loss)
                    print("Batch", batch_counter,": ", loss, " loss")

                    if batch_counter % 50 == 0:
                        torch.save(model.state_dict(), '../models_retrained_on_Hirst_data/r_' + pollen_type + '_b_' + str(batch_counter) + '.pth')
                        with open('../models_retrained_on_Hirst_data/r_' + pollen_type + '_loss.txt', "wb") as fp:
                            pickle.dump(loss_list, fp)

                        plt.plot(range(batch_counter), loss_list)
                        plt.show()
                        plt.close()

                    batch = []
                    timestamps = []
except KeyboardInterrupt:
    print("Training stopped by user.")

import os
import pandas as pd
import numpy as np
from math import factorial
from datetime import datetime, timedelta
import pickle
import torch
from torch import nn
from sklearn import preprocessing
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

from Libraries import processing as cnv
from Libraries import raw_lib as ral
RAL = ral.RAW_LIB_CLASS()


# defining parameters for training
model_version = 'model_pollen_types_ver0'      # folder name which contains the model you want to train
start_date = '201903200000'                     # data for training : year/month/day/hours/minutes
end_date = '201905200559'
pollen_type = 'BETU'                            # uppercase first four letters of pollen type (check in Hirst file if not sure)

# defining necessary functions
class preprocess:
    def normalize_image(opd, cut, normalize=False, smooth=True):
        if len(opd['Scattering']) > 3:
            image = np.asarray(opd['Scattering'])
        else:
            image = np.asarray(opd['Scattering']['Image'])
        image = np.reshape(image, [-1, 24])
        im = np.zeros([2000, 20])
        cut = int(cut)
        im_x = np.sum(image, axis=1) / 256
        N = len(im_x)

        if (im_x > 0).any():
            if N < 450:
                cm_x = 0
                for i in range(N):
                    cm_x += im_x[i] * i
                cm_x /= im_x.sum()
                cm_x = int(cm_x)
                im[1000 - cm_x:1000 + (N - cm_x), :] = image[:, 2:22]
                im = im[1000 - cut:1000 + cut, :]
                if smooth == True:
                    for i in range(20):
                        im[:, i] = preprocess.savitzky_golay(im[:, i] ** 0.5, 5, 3)
                # im[:,0:2] = 0
                # im[:,22:24] = 0
                im = np.transpose(im)
                if normalize == True:
                    return np.asarray(im / im.sum())
                else:
                    return np.asarray(im)


    def __fit_exp_decay__(x, a, b, c):
        x = np.array(x, dtype=np.float64)
        return np.array(a * np.exp(-x / b) + c, dtype=np.float64)


    def __fit_exp_approx_decay__(x, a, b, c):
        x = np.array(x, dtype=np.float64)
        return np.array(a * (
                    1.0 - (1 / b) * x + 1 / (b * 2.0) * x ** 2 - 1 / (b * 6.0) * x ** 3 + 1 / (b * 24.0) * x ** 4 - 1 / (
                        b * 120.0) * x ** 5 + 1 / (b * 720.0) * x ** 6) + c, dtype=np.float64)


    ###reconstruction of liti per band
    def __reconstruct_liti__(liti):
        indices_max = np.argwhere(liti == np.amax(liti))

        if (np.amax(liti) == 4088):

            if len(indices_max) > 1:
                ind_f = indices_max[0][0]
                ind_l = indices_max[-1][0]

                if (ind_f < 10) or (ind_l > (len(liti) - 10)):
                    liti[0::] = 0
                    return liti
                else:
                    # gauss_extrapolation
                    x1 = np.arange(0, ind_f + 1)
                    s1 = InterpolatedUnivariateSpline(x1, liti[0:ind_f + 1].astype("float64"), k=2)
                    gauss_exc_extrapolate = s1(np.arange(ind_f, ind_l + 1))

                    # exp_extrapolation
                    x2 = np.arange(0, len(liti) - ind_l)
                    popt, pcov = curve_fit(preprocess.__fit_exp_approx_decay__, x2 + (ind_l - ind_f + 1), liti[ind_l::].astype("float64"), maxfev=1000)
                    exp_decay_extrapolate = preprocess.__fit_exp_approx_decay__(np.arange(0, ind_l - ind_f + 1), popt[0], popt[1], popt[2])

                    # signal creation
                    ind_cross = np.argmin(np.abs(exp_decay_extrapolate - gauss_exc_extrapolate))
                    middle_signal = np.concatenate((gauss_exc_extrapolate[0:ind_cross], exp_decay_extrapolate[ind_cross::]))
                    liti_new = np.concatenate((liti[0:ind_f], middle_signal, liti[ind_l + 1::]))

                    return liti_new
            else:
                return liti
        else:
            return liti


    def normalize_lifitime(opd, normalize=True):
        liti = np.asarray(opd['Lifetime']).reshape(-1, 64)
        counter = 0

        for i in range(liti.shape[0]):
            if (np.amax(liti[i]) == 4088):
                liti[i] = preprocess.__reconstruct_liti__(liti[i])

            if (liti[i] == 0).all():
                counter += 1

        if counter == 0:
            lt_im = np.zeros((4, 24))
            liti_low = np.sum(liti, axis=0)
            maxx = np.max(liti_low)
            ind = np.argmax(liti_low)

            if (ind > 10) and (ind < 44):
                lt_im[:, :] = liti[:, ind - 4:20 + ind]
                weights = []
                for i in range(4):
                    weights.append(np.sum(liti[i, ind - 4:12 + ind]) - np.sum(liti[i, 0:16]))
                B = np.asarray(weights)
                A = lt_im
                if (normalize == True):
                    if (maxx > 0) and (B.max() > 0):
                        return A / maxx, B / B.max()
                else:
                    return A, B


    def savitzky_golay(y, window_size, order, deriv=0, rate=1):
        try:
            window_size = np.abs(np.int(window_size))
            order = np.abs(np.int(order))
        except ValueError:
            raise ValueError("window_size and order have to be of type int")
        if window_size % 2 != 1 or window_size < 1:
            raise TypeError("window_size size must be a positive odd number")
        if window_size < order + 2:
            raise TypeError("window_size is too small for the polynomials order")
        order_range = range(order + 1)
        half_window = (window_size - 1) // 2
        b = np.mat([[k ** i for i in order_range] for k in range(-half_window, half_window + 1)])
        m = np.linalg.pinv(b).A[deriv] * rate ** deriv * factorial(deriv)
        firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
        lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
        y = np.concatenate((firstvals, y, lastvals))
        return np.convolve(m[::-1], y, mode='valid')


    def spec_correction(opd, normalize=True):
        spec = np.asarray(opd['Spectrometer'])
        spec_2D = spec.reshape(-1, 8)

        b = 0
        if (spec_2D[:, 1] > 20000).any():
            res_spec = spec_2D[:, 1:5]
            b = 1
        else:
            res_spec = spec_2D[:, 0:4]

        if (np.argsort(res_spec[:, 0])[-4:] > 3).all() and (np.argsort(res_spec[:, 0])[-4:] < 10).all():

            if b == 0:
                res_spec = spec_2D[:, 1:5]

            for i in range(res_spec.shape[1]):
                res_spec[:, i] -= np.minimum(spec_2D[:, 6], spec_2D[:, 7])

            for i in range(4):
                res_spec[:, i] = preprocess.savitzky_golay(res_spec[:, i], 5, 3)  # Spectrum is smoothed
            res_spec = np.transpose(res_spec)
            if normalize == True:
                A = res_spec
                if (A.max() > 0):
                    return A / A.max()
            else:
                return res_spec


    def size_particle(opd, scale_factor=1):
        image = np.asarray(opd['Scattering']['Image']).reshape(-1, 24)
        x = (np.asarray(image, dtype='float32').reshape(-1, 24)[:, :]).sum()
        if x < 5500000:
            return 0.5
        elif (x >= 5500000) and (x < 500000000):
            return 9.95e-01 * np.log(3.81e-05 * x) - 4.84e+00
        else:
            return 0.0004 * x ** 0.5 - 3.9                             #
def extract_features(data):
    spectrum_list = []
    scatter_list = []
    size_list = []
    lifetime_list1 = []
    lifetime_list2 = []

    for i in range(len(data)):

        if np.max(data[i]["Spectrometer"]) > 2500:
            scat = preprocess.normalize_image(data[i], cut=60, normalize=False, smooth=True)
            life = preprocess.normalize_lifitime(data[i], normalize=True)
            spec = preprocess.spec_correction(data[i], normalize=True)
            size = preprocess.size_particle(data[i], scale_factor=1)

            if scat is not None and spec is not None and life is not None:
                scatter_list.append(scat)
                size_list.append(size)
                lifetime_list1.append(life[0])
                spectrum_list.append(spec)
                lifetime_list2.append(life[1])

    return scatter_list, spectrum_list, lifetime_list1, lifetime_list2, size_list
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
   
# paths to data and model
raw_df_path = '../data/novi_sad/DATA'
path = 'models/novi_sad/' + model_version + '/'
params_files_list = [file[:-4] for file in os.listdir(path)]
params_list = [pollen_type, 'Other']
number_of_classes = len(params_list)

# import Hirst data and define moments of calibrations not to use while training
hirst = pd.read_excel("Libraries/HIRST_2019021601_2019101706.xlsx")
calibs = ["2019-02-07 14", "2019-02-07 16", "2019-02-11 08", "2019-02-18 16", "2019-02-18 17", "2019-02-18 18", "2019-02-25 12", "2019-02-25 13",
          "2019-03-11 12", "2019-03-11 13", "2019-03-15 13", "2019-03-15 14", "2019-03-19 07", "2019-03-19 08", "2019-03-19 09", "2019-03-25 08",
          "2019-03-25 09", "2019-03-25 10", "2019-03-26 08",  "2019-03-26 09", "2019-03-29 08", "2019-03-29 09", "2019-04-04 10", "2019-04-04 11",
          "2019-04-04 12", "2019-04-09 07", "2019-04-09 08", "2019-04-15 07", "2019-04-15 08", "2019-04-18 16" , "2019-04-22 08", "2019-04-22 09",
          "2019-04-22 14", "2019-04-22 15", "2019-04-29 10", "2019-04-29 11", "2019-05-03 08", "2019-05-03 09", "2019-05-07 16", "2019-05-07 17",
          "2019-05-10 16", "2019-05-10 17", "2019-05-24 13", "2019-05-24 14", "2019-06-03 11", "2019-06-03 12", "2019-06-13 15"]

# get a list of folders which contain data
folders = os.listdir(raw_df_path)
folders = sorted(folders)
print(folders)

# define and load model
model = Net(number_of_classes)
model.load_state_dict(torch.load(path + params_files_list[0] + '.pth', map_location=lambda storage, loc: storage))
model.eval()


batch_size = 24         # the size (number of hours) of one batch
batch, timestamps, loss_list, scatter_list, spectrum_list, lifetime_list1, lifetime_list2, size_list = [], [], [], [], [], [], [], []
batch_counter = 0
while True:
    for folder in folders:          # go through folders
        files = os.listdir("%s/%s" % (raw_df_path, folder))         # get files in each folder
        sorted_files = sorted(files, key=lambda x: int(x.split('_')[2][:-4]), reverse=False)

        for file in sorted_files:
            dt = file.split('_')[2][:-4]           # file name contains datetime
            minutes = int(dt[10:12])

            if (dt >= start_date) and (dt <= end_date):
                try:
                    converter = cnv.convertion("%s/%s/%s" % (raw_df_path, folder, file))            # get data
                    raw_data = converter.global_dic()['Data']
                except:
                    print('except', file)
                    continue

                # collect dsta for one hour
                scat, spec, life1, life2, size = extract_features(raw_data)
                scatter_list += scat
                spectrum_list += spec
                lifetime_list1 += life1
                lifetime_list2 += life2
                size_list += size

                if minutes == 59:       # if it is the end of an hour
                    str_date = datetime.strptime('%s-%s-%s %s' % (dt[0:4], dt[4:6], dt[6:8], dt[8:10]), '%Y-%m-%d %H')
                    timestamp = datetime.strftime(str_date + timedelta(hours=1), "%Y-%m-%d %H:00:00+00:00")
                    print(timestamp, len(scatter_list))

                    if timestamp[:13] not in calibs and len(scatter_list) != 0:         #if this hour is not in calibrations

                        # add collected data as one sample in batch list
                        timestamps.append(timestamp)
                        batch.append([scatter_list, spectrum_list, lifetime_list1, lifetime_list2, size_list])
                        scatter_list, spectrum_list, lifetime_list1, lifetime_list2, size_list = [], [], [], [], []

                # CODE FOR RETRAINING
                if len(batch) == batch_size:
                    print("Training on collected batch ", batch_counter)

                    # GETTING REAL VALUES FROM HIRST
                    timestamp_indices = np.array([np.where(hirst['Unnamed: 0'] == timestamps[i])[0][0] for i in range(batch_size)])
                    #y_true = np.zeros((batch_size))
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

                    # CREATE LISTS CONTAINING INPUT TENSORS FOR NETWORK FOR EACH SAMPLE (HOUR) IN BATCH
                    spectrum_tensor, scatter_tensor, lifetime_tensor1, lifetime_tensor2, size_tensor = [], [], [], [], []
                    for i in range(batch_size):
                        if len(batch[i][1]) != 0:
                            spectrum_tensor.append(torch.Tensor(batch[i][1]).unsqueeze_(0).permute(1, 0, 2, 3))
                            scatter_tensor.append(torch.Tensor(batch[i][0]).unsqueeze_(0).permute(1, 0, 2, 3))
                            lifetime_tensor1.append(torch.Tensor(batch[i][2]).unsqueeze_(0).permute(1, 0, 2, 3))
                            lifetime_tensor2.append(torch.Tensor(batch[i][3]))
                            size_tensor.append(torch.Tensor(batch[i][4]).unsqueeze_(0).permute(1, 0))

                    output = model(spectrum_tensor, scatter_tensor, lifetime_tensor1, lifetime_tensor2, size_tensor, var)

                    # DEFINE OPTIMIZER AND LOSS FUNCTION
                    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
                    loss_fn = torch.nn.MSELoss()

                    loss = loss_fn(output, y_true_tensor) # CALCULATE LOSS
                    print("Loss: ", loss)
                    loss_list.append(loss)

                    optimizer.zero_grad()   # CLEAN GRADIENTS FOR NEXT TRAIN
                    loss.backward()         # BACKPROP, COMPUTE GRADIENTS
                    optimizer.step()        # APPLY GRADIENTS

                    # SAVE MODEL AND LOSS LIST
                    torch.save(model.state_dict(), '../models_retrained_on_Hirst_data/r_' + model_version + '_season.pth')
                    with open('../models_retrained_on_Hirst_data/r_' + model_version + '_season_loss.txt', "wb") as fp:
                        pickle.dump(loss_list, fp)

                    #if batch_counter%10 == 0:
                     #   plt.plot(loss_list)
                      #  plt.show()

                    batch = []
                    timestamps = []
                    batch_counter += 1
                    print("***************************************************************************************************")

                    # for parameter in model.parameters():
                    #   print(parameter)

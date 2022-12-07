import os
import numpy as np
from math import factorial
import torch
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.optimize import curve_fit
import json


from Libraries import processing as cnv
from Libraries import raw_lib as ral

RAL = ral.RAW_LIB_CLASS()

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
                    popt, pcov = curve_fit(preprocess.__fit_exp_approx_decay__, x2 + (ind_l - ind_f + 1),
                                           liti[ind_l::].astype("float64"), maxfev=1000)
                    exp_decay_extrapolate = preprocess.__fit_exp_approx_decay__(np.arange(0, ind_l - ind_f + 1),
                                                                                popt[0], popt[1], popt[2])

                    # signal creation
                    ind_cross = np.argmin(np.abs(exp_decay_extrapolate - gauss_exc_extrapolate))
                    middle_signal = np.concatenate(
                        (gauss_exc_extrapolate[0:ind_cross], exp_decay_extrapolate[ind_cross::]))
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
            return 0.0004 * x ** 0.5 - 3.9  #


def extract_features(data):
    spectrum_list = []
    scatter_list = []
    size_list = []
    lifetime_list1 = []
    lifetime_list2 = []
    selected_part_ind = []
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
                selected_part_ind.append(i)

    return selected_part_ind, scatter_list, spectrum_list, lifetime_list1, lifetime_list2, size_list


dir_name = '../data/calib_data_ns'
target_dir = '../data/calib_data_ns_tensors'
ddf = {'columns' : ['Filename', 'Timestamp', 'Label']}
dir = os.listdir(dir_name)
labels_names = []
labels = []
fnames = []
timestamps = []
lab = 0;
for fn in dir:
    with open(os.path.join(dir_name,fn), 'r') as f:
        data = json.load(f)
        list_ind, scat, spec, life1, life2, size = extract_features(data)
        for idx in range(len(list_ind)):
            scat_t = torch.Tensor(scat[idx]).view(1,20,120)
            spec_t = torch.Tensor(spec[idx]).view(1,4,32)
            lif1_t = torch.Tensor(life1[idx]).view(1, 4, 24)
            lif2_t = torch.Tensor(life2[idx]).view(4)
            size_t = torch.Tensor([size[idx]]).view(1)

            dd = {'Timestamp': data[list_ind[idx]]['Timestamp'],
                  'Scatter': scat_t,
                  'Spectrum': spec_t,
                  'Lifetime 1': lif1_t,
                  'Lifetime 2': lif2_t,
                  'Size': size_t,
                  'Label': lab
                  }
            fname = (fn[:-5] + '_' + str(data[list_ind[idx]]['Timestamp']) + '.pt').replace(' ','_').replace(':','_')
            fnames.append(fname)
            timestamps.append(data[list_ind[idx]]['Timestamp'])
            labels.append(lab)
            torch.save(dd,os.path.join(target_dir,fname))
    lab += 1
    labels_names.append(fn[:-5])
ddf['type'] = 'train'
ddf['num_of_samples'] = len(fnames)
ddf['mean'] = None
ddf['std'] = None
ddf['min'] = None
ddf['max'] = None
ddf['pair_split_path'] = None
ddf['parent'] = None
ddf['train_splits']=[]
ddf['test_splits'] = []
ddf['data'] = [fnames,timestamps,labels]
torch.save(ddf,'/home/guest/coderepos/transfer_learning/Rapid-E/meta_jsons/00000_train.pt')
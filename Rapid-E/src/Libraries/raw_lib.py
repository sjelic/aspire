#####################################################################
# aim of this script is to have a class                  			#
# that gives you three main features preprocessed and converted to  #
# a fixed format													#
#####################################################################

import numpy as np
from math import factorial

class RAW_LIB_CLASS():

    def __init__(self):
        pass

    #################################################################################################
    # Smooth function
    #################################################################################################
    def smooth(self, y, window_size, order, deriv=0, rate=1):
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


    def raw2processed_scat(self, opd, cut=60, normalize=False, smooth=True):
        if len(opd['Scattering']) > 3:
            image = np.asarray(opd['Scattering'])
        else:
            image = np.asarray(opd['Scattering']['Image'])
        image = np.reshape(image, [-1, 24])
        im = np.zeros([2000, 24])
        cut = int(cut)
        im_x = np.sum(image, axis=1) / 256
        N = len(im_x)
        if N > 450:
            return 0,image #np.zeros([cut * 2, 24])
        N2 = int(N / 2)
        cm_x = 0
        for i in range(N):
            cm_x += im_x[i] * i
        cm_x /= im_x.sum()
        cm_x = int(cm_x)
        im[1000 - cm_x:1000 + (N - cm_x), :] = image[:, :]
        im = im[1000 - cut:1000 + cut, :]
        if smooth == True:
            for i in range(2, 22):
                im[:, i] = self.smooth(im[:, i] ** 0.5, 5, 3)
        im[:, 0:2] = 0
        im[:, 22:24] = 0
        if normalize == True:
            return 1, np.asarray(im / im.sum())
        else:
            return 1, np.asarray(im)

    #################################################################################################
    # Lifetime processing
    # Returns a vector of 24 values = sum over all channels - least significant (for noise reduction)
    # Returns also normalized to max over all integrals of four channels
    # If bad signal, returns zeros
    #################################################################################################
    def raw2processed_liti(self, opd, normalize=True):
        liti = np.asarray(opd['Lifetime']).reshape(-1, 64)
        liti_low = np.sum(liti, axis=0)
        ind = np.argmax(liti_low)
        if (ind < 10) or (ind > 44):
            return 0, np.zeros(28)
        else:
            liti_low[0:24] = liti_low[ind - 4:20 + ind]
            weights = []
            for i in range(4):
                weights.append(np.sum(liti[i, ind - 4:12 + ind]) - np.sum(liti[i, 0:16]))
            A = np.asarray(liti_low[0:24])
            B = np.asarray(weights)
            if (normalize == True):
                if (A.max() > 0) and (B.max() > 0):
                    return 1, np.concatenate((A / A.max(), B), axis=0)
                else:
                    return 0, np.concatenate((A,B),axis = 0)#np.zeros(28)
            else:
                return 1, np.concatenate((A, B), axis=0)


    #################################################################################################
    # Fluorescence spectrum processing
    # Returns a vector of 32 values = sum over 2, 3, 4 and 5th acquisitions - last or before last acquisition (for noise reduction)
    # Fluorescence spectrum is only used in normalized shape
    # If bad signal, returns zeros
    #################################################################################################
    def raw2processed_fluo(self, opd, normalize=True):
        spec = np.asarray(opd['Spectrometer'])
        spec_2D = spec.reshape(-1, 8)
        if (spec_2D[:, 1] > 20000).any():
            res_spec = spec_2D[:, 1] + spec_2D[:, 2] + spec_2D[:, 3] + spec_2D[:, 4] - 4 * np.minimum(spec_2D[:, 6],
                                                                                                      spec_2D[:,
                                                                                                      7])  # 0 acquisition is ignored since it might be saturated
        else:
            res_spec = spec_2D[:, 0] + spec_2D[:, 1] + spec_2D[:, 2] + spec_2D[:, 3] - 4 * np.minimum(spec_2D[:, 6],
                                                                                                      spec_2D[:, 7])
        if normalize == True:
            A = self.smooth(res_spec, 5, 3)  # Spectrum is smoothed
            if (A.max() > 0):
                return 1, A / A.max()  # Spectrum is normalized
            else:
                return 0, A# np.zeros(32)
        else:
            return 1, self.smooth(res_spec, 5, 3)

    #################################################################################################
    # Particle equivalent otpical size
    # Returns size estimation in micrometers (um)
    # Empirical function
    #################################################################################################
    def size(self, opd, scale_factor=1):
        image = np.asarray(opd['Scattering']['Image']).reshape(-1, 24)
        x = (np.asarray(image, dtype='float32').reshape(-1, 24)[:, :]).sum()
        if x < 5500000:
            return 0.5
        elif (x >= 5500000) and (x < 500000000):
            return 9.95e-01 * np.log(3.81e-05 * x) - 4.84e+00
        else:
            return 0.0004 * x ** 0.5 - 3.9
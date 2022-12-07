# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 00:42:33 2020

@author: sjelic
"""

import os
# import pandas as pd
import numpy as np
import pickle
import torch
from torch.utils.data import Dataset
import pandas as pd
from train_test_splits import load_meta_json

base_dir = '/home/guest/coderepos/transfer_learning/Rapid-E'
tensor_subdir = 'data/calib_data_ns_tensors'
meta_subdir = 'meta_jsons'


class RapidEDataset(Dataset):
    """RapidE dataset."""

    def __init__(self, df, dir_path, df_pollen_info, load=False, name='dataset', preloaded_dict=None):
        """
        Args:
            pollen_type - one of 26 possible pollen types
            df (string): pandas frame with metada and .
            dir_path (string): Directory with all the data.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df = df.copy()
        self.dir_path = dir_path
        self.name = name
        self.df_pollen_info = df_pollen_info
        self.pollen_types = list(df.columns[4:])
        self.num_of_classes = len(self.pollen_types)
        self.preloaded_dict = preloaded_dict
        self.load = load

        self.monts_of_types = df_pollen_info[['START', 'END']][df_pollen_info['CODE'].isin(self.pollen_types)]

        self.monts_of_types = self.monts_of_types.set_index(pd.Index(list(range(len(self.monts_of_types)))))
        for i in range(len(self.pollen_types)):
            inseas = sum(list(map(lambda x: 1 if (
                        x >= self.monts_of_types.loc[i, 'START'] and x <= self.monts_of_types.loc[i, 'START']) else 0,
                                  list(self.df['MONTH']))))
            outseas = len(self.df) - inseas
            weights = list(map(lambda x: 1.0 / (2 * inseas) if (
                        x >= self.monts_of_types.loc[i, 'START'] and x <= self.monts_of_types.loc[
                    i, 'START']) else 1.0 / (2 * outseas), list(self.df['MONTH'])))
            self.df[self.pollen_types[i] + '_W'] = weights

        # self.transform = transform

    def load_to_torch_tensor(self):
        fnames = list(self.df['FILENAME'])
        for fn in fnames:
            with open(os.path.join(self.dir_path, fn), 'rb') as file:
                X = pickle.load(file)
                X[0] = torch.Tensor(X[0]).unsqueeze_(0).permute(1, 0, 2, 3)
                X[1] = torch.Tensor(X[1]).unsqueeze_(0).permute(1, 0, 2, 3)
                X[2] = torch.Tensor(X[2]).unsqueeze_(0).permute(1, 0, 2, 3)
                X[3] = torch.Tensor(X[3])
                X[4] = torch.Tensor(X[4]).unsqueeze_(0).permute(1, 0)
                self.dataset.append(X)

        # y = torch.tensor(np.array(list(self.df.iloc[idx,4:(4+self.num_of_classes)])),dtype=torch.float32)
        # w = torch.tensor(np.array(list(self.df.iloc[idx,(4+self.num_of_classes):])),dtype=torch.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.load:
            X = self.preloaded_dict[self.df.loc[idx, 'FILENAME']]
            y = torch.tensor(np.array(list(self.df.iloc[idx, 4:(4 + self.num_of_classes)])), dtype=torch.float32)
            w = torch.tensor(np.array(list(self.df.iloc[idx, (4 + self.num_of_classes):])), dtype=torch.float32)

            # data = [[X[i],y[i],w[i]] for i in range(len(X))]

        else:
            with open(os.path.join(self.dir_path, self.df.loc[idx, 'FILENAME']), 'rb') as file:
                # print(idx)
                X = pickle.load(file)
                X[0] = torch.Tensor(X[0]).unsqueeze_(0).permute(1, 0, 2, 3)
                X[1] = torch.Tensor(X[1]).unsqueeze_(0).permute(1, 0, 2, 3)
                X[2] = torch.Tensor(X[2]).unsqueeze_(0).permute(1, 0, 2, 3)
                X[3] = torch.Tensor(X[3])
                X[4] = torch.Tensor(X[4]).unsqueeze_(0).permute(1, 0)

                y = torch.Tensor(list(self.df.iloc[idx, 4:(4 + self.num_of_classes)]) * (X[0].shape[0])).reshape(-1, 1)
                w = torch.Tensor(list(self.df.iloc[idx, (4 + self.num_of_classes):]) * (X[0].shape[0])).reshape(-1, 1)
                X.append(y)
                X.append(w)

                print(X[0].shape)
                print(X[1].shape)
                print(X[2].shape)
                print(X[3].shape)
                print(X[4].shape)
                print(X[5].shape)
                print(X[6].shape)
                # print(len(X))
                # data = [[X[i],y[i],w[i]] for i in range(len(X))]
            # print(y)
            # print(y)
            # w = np.array(list(self.df.iloc[idx,(4+len(self.pollen_types)):]))

        return X

    def save_to_file(self):
        self.df.to_excel('probni.xlsx')


class Standardize(object):
    def __init__(self, meta_json):
        self.dd = torch.load(meta_json)
        self.dd['std']['Scatter'][self.dd['std']['Scatter'] == 0] = 1
        self.dd['std']['Spectrum'][self.dd['std']['Spectrum'] == 0] = 1
        self.dd['std']['Lifetime 1'][self.dd['std']['Lifetime 1'] == 0] = 1
        self.dd['std']['Lifetime 2'][self.dd['std']['Lifetime 2'] == 0] = 1
        self.dd['std']['Size'][self.dd['std']['Size'] == 0] = 1

    def __call__(self, ddf):
        ddf['Scatter'] = (ddf['Scatter'] - self.dd['mean']['Scatter']) / self.dd['std']['Scatter']
        ddf['Spectrum'] = (ddf['Spectrum'] - self.dd['mean']['Spectrum']) / self.dd['std']['Spectrum']
        ddf['Lifetime 1'] = (ddf['Lifetime 1'] - self.dd['mean']['Lifetime 1']) / self.dd['std']['Lifetime 1']
        ddf['Lifetime 2'] = (ddf['Lifetime 2'] - self.dd['mean']['Lifetime 2']) / self.dd['std']['Lifetime 2']
        ddf['Size'] = (ddf['Size'] - self.dd['mean']['Size']) / self.dd['std']['Size']
        return ddf


class RapidECalibrationDataset(Dataset):
    def __init__(self, meta_json, transform=None):
        self.dd = torch.load(meta_json)
        self.df = load_meta_json(meta_json)
        self.transform = transform

    def __getitem__(self, idx):
        #y = int(self.df.loc[idx, 'Label'])
        ddf = torch.load(os.path.join(base_dir, tensor_subdir, self.df.loc[idx, 'Filename']))
        if self.transform:
            ddf = self.transform(ddf)

        return ddf

    def __len__(self):
        return self.dd['num_of_samples']

dataset_meta = '/home/guest/coderepos/transfer_learning/Rapid-E/meta_jsons/00000_train.pt'
standardization = Standardize(dataset_meta)

dataset = RapidECalibrationDataset(dataset_meta, transform=standardization)
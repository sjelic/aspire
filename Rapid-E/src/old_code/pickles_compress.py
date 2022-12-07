# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 09:06:22 2020

@author: Korisnik
"""

import os
#import pandas as pd
import pickle
import torch
from utils import open_excel_file_in_pandas

df = open_excel_file_in_pandas('./Libraries/data_pandas_frame.xlsx')
dir_path = '../../data/novi_sad_2019_'

fns = list(df['FILENAME'])
dics = {}

for fn in fns:
    with open(os.path.join(dir_path, fn),'rb') as file:
        X = pickle.load(file)
        X[0] = torch.Tensor(X[0]).unsqueeze_(0).permute(1, 0, 2, 3)
        X[1] = torch.Tensor(X[1]).unsqueeze_(0).permute(1, 0, 2, 3)
        X[2] = torch.Tensor(X[2]).unsqueeze_(0).permute(1, 0, 2, 3)
        X[3] = torch.Tensor(X[3])
        X[4] = torch.Tensor(X[4]).unsqueeze_(0).permute(1, 0)
        dics[fn] = X

with open('../data/novi_sad_2019.pkl', 'wb') as handle:
    pickle.dump(dics, handle, protocol=pickle.HIGHEST_PROTOCOL)

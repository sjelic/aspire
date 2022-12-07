import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import os
import torch

base_dir = '/home/guest/coderepos/transfer_learning/Rapid-E'
tensor_subdir = 'data/calib_data_ns_tensors'
meta_subdir = 'meta_jsons'


def load_meta_json(fpath):
    dd = torch.load(fpath)
    df = pd.DataFrame(columns=dd['columns'])
    for idx, col in enumerate(dd['columns']):
        df[col] = dd['data'][idx]
    return df



def train_test_split(meta_json, num_of_splits = 10, test_size=0.2):

    if not os.path.isdir(os.path.join(base_dir,meta_subdir)):
        os.mkdir(os.path.join(base_dir,meta_subdir))
    df = load_meta_json(meta_json)
    dd = torch.load(meta_json)

    count = 0
    id = 0
    sss = StratifiedShuffleSplit(n_splits=num_of_splits, test_size=test_size , random_state=0)
    for train, test in sss.split(np.zeros(len(df)), list(df['Label'])):
        while True:
            id += 1
            train_fname = str(id).zfill(5) + '_train.pt'
            test_fname = str(id).zfill(5) + '_test.pt'
            if not os.path.isfile(os.path.join(base_dir, meta_subdir, train_fname)):
                break


        df_train = df.iloc[train].sort_values(by=['Label','Timestamp'])
        df_test = df.iloc[test].sort_values(by=['Label','Timestamp'])

        ddf_train = {'columns': ['Filename', 'Timestamp', 'Label']}
        ddf_train['type'] = 'train'
        ddf_train['num_of_samples'] = len(df_train)
        ddf_train['mean'] = None
        ddf_train['std'] = None
        ddf_train['min'] = None
        ddf_train['max'] = None
        ddf_train['train_splits'] = []
        ddf_train['test_splits'] = []
        ddf_train['pair_split_path'] = test_fname
        ddf_train['parent'] = meta_json
        ddf_train['data'] = [list(df_train['Filename']), list(df_train['Timestamp']), list(df_train['Label'])]


        ddf_test = {'columns': ['Filename', 'Timestamp', 'Label']}
        ddf_test['type'] = 'test'
        ddf_test['num_of_samples'] = len(df_test)
        ddf_test['mean'] = None
        ddf_test['std'] = None
        ddf_test['min'] = None
        ddf_test['max'] = None
        ddf_test['train_splits'] = []
        ddf_test['test_splits'] = []
        ddf_test['pair_split_path'] = train_fname
        ddf_test['parent'] = meta_json
        ddf_test['data'] = [list(df_test['Filename']), list(df_test['Timestamp']), list(df_test['Label'])]

        torch.save(ddf_train,os.path.join(base_dir, meta_subdir, train_fname))
        torch.save(ddf_test, os.path.join(base_dir, meta_subdir, test_fname))

        dd['train_splits'].append(train_fname)
        dd['test_splits'].append(test_fname)

    torch.save(dd,meta_json)

        
def train_valid_test_split(meta_json, num_of_splits = 10, test_size=0.2):
    train_test_split(meta_json, num_of_splits=num_of_splits, test_size=test_size)
    dd = torch.load(meta_json)
    for fn in dd['train_splits']:
        train_test_split(os.path.join(base_dir,meta_subdir,fn), num_of_splits=num_of_splits, test_size=test_size)


def update_mean_max_min(meta_json):
    dd = torch.load(meta_json)
    if (dd['mean'] is not None) or (dd['min'] is not None) or (dd['max'] is not None):
        return
    elif dd['type'] == 'test':
        update_mean_max_min(os.path.join(base_dir,meta_subdir,dd['pair_split_path']))
        ddt = torch.load(os.path.join(base_dir,meta_subdir,dd['pair_split_path']))
        dd['mean'] = ddt['mean']
        dd['min'] = ddt['min']
        dd['max'] = ddt['max']
    else:
        mean = {'Scatter': torch.zeros(1,20,120),
                'Spectrum': torch.zeros(1,4,32),
                'Lifetime 1': torch.zeros(1,4,24),
                'Lifetime 2': torch.zeros(4),
                'Size': torch.zeros(1)}
        min = {'Scatter': 1e20 * torch.ones(1,20,120),
               'Spectrum': 1e20 * torch.ones(1,4,32),
               'Lifetime 1': 1e20 * torch.ones(1,4,24),
               'Lifetime 2': 1e20 * torch.ones(4),
               'Size': 1e20 * torch.ones(1)}
        max = {'Scatter': -1e20 * torch.ones(1, 20, 120),
               'Spectrum': -1e20 * torch.ones(1, 4, 32),
               'Lifetime 1': -1e20 * torch.ones(1, 4, 24),
               'Lifetime 2': -1e20 * torch.ones(4),
               'Size': -1e20 * torch.ones(1)}

        df = load_meta_json(meta_json)

        for idx in range(len(df)):
            # print('Reading file: ' +  os.path.join(base_dir,image_subdir,df.loc[idx,'image_path']))
            ddf = torch.load(os.path.join(base_dir,tensor_subdir,df.loc[idx,'Filename']))
            for key in mean.keys():
                mean[key] += ddf[key]
                min[key] = torch.min(min[key],ddf[key])
                max[key] = torch.max(max[key], ddf[key])

        for key in mean.keys():
            mean[key] /= dd['num_of_samples']

        dd['mean'] = mean
        dd['min'] = min
        dd['max'] = max
    torch.save(dd,meta_json)

def update_std(meta_json):
    update_mean_max_min(meta_json)
    dd = torch.load(meta_json)
    if dd['std'] is not None:
        return
    elif dd['type'] == 'test':
        update_std(os.path.join(base_dir, meta_subdir, dd['pair_split_path']))
        ddt = torch.load(os.path.join(base_dir, meta_subdir, dd['pair_split_path']))
        dd['std'] = ddt['std']
    else:
        std = {'Scatter': torch.zeros( 1, 20, 120),
                'Spectrum': torch.zeros(1, 4, 32),
                'Lifetime 1': torch.zeros( 1, 4, 24),
                'Lifetime 2': torch.zeros(4),
                'Size': torch.zeros(1)}
        df = load_meta_json(meta_json)
        for idx in range(len(df)):
            # print('Reading file: ' +  os.path.join(base_dir,image_subdir,df.loc[idx,'image_path']))
            ddf = torch.load(os.path.join(base_dir,tensor_subdir,df.loc[idx,'Filename']))
            for key in std.keys():
                std[key] += (ddf[key] - dd['mean'][key])**2

        for key in std.keys():
            std[key] = torch.sqrt(std[key]/dd['num_of_samples'])

        dd['std'] = std
    torch.save(dd, meta_json)

def update_statistics():
    meta_json_list = os.listdir(os.path.join(base_dir,meta_subdir))
    for fn in meta_json_list:
        update_mean_max_min(os.path.join(base_dir,meta_subdir,fn))
        update_std(os.path.join(base_dir, meta_subdir, fn))
        print(fn + ' finished.')



#train_valid_test_split('/home/guest/coderepos/transfer_learning/Rapid-E/meta_jsons/00000_train.pt')
#update_statistics()




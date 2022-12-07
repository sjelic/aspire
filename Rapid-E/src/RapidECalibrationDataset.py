import torch
import pandas as pd
import os
from torch.utils.data import Dataset

def load_meta_json(fpath):
    dd = torch.load(fpath)
    df = pd.DataFrame(columns=dd['columns'])
    for idx, col in enumerate(dd['columns']):
        df[col] = dd['data'][idx]
    return df

class RapidECalibrationDataset(Dataset):
    def __init__(self, meta_json, transform=None):
        self.meta_json = meta_json
        self.dd = torch.load(meta_json)
        self.df = load_meta_json(meta_json)
        self.transform = transform
        self.train_splits = self.dd['train_splits']
        self.name = os.path.splitext(os.path.basename(meta_json))[0]


    def __getitem__(self, idx):
        #y = int(self.df.loc[idx, 'Label'])
        # os.path.join(base_dir, tensor_subdir, self.df.loc[idx, 'Filename'])
        ddf = torch.load(self.df.loc[idx, 'Filename']) # Filepath, please correct this
        if self.transform:
            ddf = self.transform(ddf)
        return ddf

    def __len__(self):
        return self.dd['num_of_samples']


    def gettargets(self):
        return self.dd['data'][2]
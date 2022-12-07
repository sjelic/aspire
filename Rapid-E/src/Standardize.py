import torch

class RapidECalibrationStandardize(object):
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

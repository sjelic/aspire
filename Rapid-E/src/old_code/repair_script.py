import torch
import os

list_dir = os.listdir('/homeq/guest/coderepos/transfer_learning/Rapid-E/meta_jsons')
for fn in list_dir:
    dd = torch.load(os.path.join('/home/guest/coderepos/transfer_learning/Rapid-E/meta_jsons',fn))
    dd['mean']['Scatter'] = dd['mean']['Scatter'][0]
    dd['mean']['Spectrum'] = dd['mean']['Spectrum'][0]
    dd['mean']['Lifetime 1'] = dd['mean']['Lifetime 1'][0]
    dd['mean']['Lifetime 2'] = dd['mean']['Lifetime 2'][0]
    dd['mean']['Size'] = dd['mean']['Size'][0]

    dd['std']['Scatter'] = dd['std']['Scatter'][0]
    dd['std']['Spectrum'] = dd['std']['Spectrum'][0]
    dd['std']['Lifetime 1'] = dd['std']['Lifetime 1'][0]
    dd['std']['Lifetime 2'] = dd['std']['Lifetime 2'][0]
    dd['std']['Size'] = dd['std']['Size'][0]

    dd['min']['Scatter'] = dd['min']['Scatter'][0]
    dd['min']['Spectrum'] = dd['min']['Spectrum'][0]
    dd['min']['Lifetime 1'] = dd['min']['Lifetime 1'][0]
    dd['min']['Lifetime 2'] = dd['min']['Lifetime 2'][0]
    dd['min']['Size'] = dd['min']['Size'][0]

    dd['max']['Scatter'] = dd['max']['Scatter'][0]
    dd['max']['Spectrum'] = dd['max']['Spectrum'][0]
    dd['max']['Lifetime 1'] = dd['max']['Lifetime 1'][0]
    dd['max']['Lifetime 2'] = dd['max']['Lifetime 2'][0]
    dd['max']['Size'] = dd['max']['Size'][0]

    torch.save(dd,os.path.join('/home/guest/coderepos/transfer_learning/Rapid-E/meta_jsons',fn))

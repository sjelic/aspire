import os
import torch
# dirp = '/home/guest/coderepos/transfer_learning/Rapid-E/data/calib_data_ns_tensors'
# files = os.listdir(dirp)
#
# for file in files:
#     dd = torch.load(os.path.join(dirp,file))
#     dd['Target'] = dd['Label']
#     del dd['Label']
#     torch.save(dd,os.path.join(dirp,file))
    


#file = 'Plantago_2018-06-15_09_39_28260.pt'
# dd = torch.load(os.path.join(dirp,file))


dirp2 = '/home/guest/coderepos/transfer_learning/Rapid-E/meta_jsons'
files2 = os.listdir(dirp2)
for file in files2:
    dd = torch.load(os.path.join(dirp2,file))
    dd['columns'] = ['Filename', 'Timestamp', 'Target']
    if dd['pair_split_path'] is not None:
        dd['pair_split_path'] = '../meta_jsons/' + dd['pair_split_path']
    dd['train_splits'] = ['../meta_jsons/' + fn for fn in dd['train_splits']]
    dd['test_splits'] = ['../meta_jsons/' + fn for fn in dd['test_splits']]
    if dd['parent'] is not None:
        dd['parent'] = os.path.relpath(dd['parent'],'/home/guest/coderepos/transfer_learning/Rapid-E/src')
    dd['data'][0] = ['../data/calib_data_ns_tensors/' + fn for fn in dd['data'][0]]
    torch.save(dd,os.path.join(dirp2,file))

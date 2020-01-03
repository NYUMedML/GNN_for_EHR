from torch.utils.data import Dataset
import pickle
import numpy as np
import torch


def load_pickle(fname):
    with open(fname, 'rb') as f:  
        return pickle.load(f)



def downsample(train_idx, neg_young, train_idx_pos):
    downsamples = np.random.permutation(neg_young)[:450000]
    mask=np.ones(len(train_idx), bool)
    mask[downsamples] = False
    downsample_idx = np.concatenate((train_idx[mask], np.repeat(train_idx_pos,50)))
    return downsample_idx


class OriginalData:
    def __init__(self, path):
        self.path = path
        self.feature_selection = load_pickle(path + 'frts_selection.pkl')
        self.x = load_pickle(path + 'preprocess_x.pkl')[:, self.feature_selection]
        self.y = load_pickle(path + 'y_bin.pkl')
        
    def datasampler(self, idx_path, train = True):
        idx = load_pickle(self.path + idx_path)
        if train:
            downsample_idx = downsample(idx, load_pickle(self.path + 'neg_young.pkl'), idx[self.y[idx] == 1])
            return self.x[downsample_idx, :], self.y[downsample_idx]
        return self.x, self.y


class EHRData(Dataset):
    def __init__(self, data, cla):
        self.data = data
        self.cla = cla
        
    def __len__(self):
        return len(self.cla)
        
    def __getitem__(self, idx):
        return self.data[idx], self.cla[idx]


def collate_fn(data):
    # padding
    data_list = []
    for datum in data:
        data_list.append(np.hstack((datum[0].toarray().ravel(), datum[1])))
    return torch.from_numpy(np.array(data_list)).long()
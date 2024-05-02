import numpy as np
import torch
from torch.utils.data import Dataset
import h5py

class H5Dataset(Dataset):
    def __init__(self, h5_filename, LSTM_step, len_caorse, len_dense):
        self.h5_filename = h5_filename
        self.h5_file = h5py.File(h5_filename, 'r')
        self.groups = list(self.h5_file.keys())
        self.step = LSTM_step
        self.len_coarse, self.len_dense = len_caorse, len_dense

        self.tot_len_coarse = int(LSTM_step*len_caorse)
        self.tot_len_dense = int(LSTM_step*len_dense)

    def __len__(self):
        return len(self.groups)
    
    def log_normalize(self, x):
        return (20*np.log10(np.abs(x))+50)/50

    def __getitem__(self, idx):
        if not self.h5_file.id:
            self.h5_file = h5py.File(self.h5_filename, 'r')

        group = self.h5_file[self.groups[idx]]

        pattern = group['pattern'][:]/10
        sub_thickness = group['sub_thickness'][()]*10
        sub_info = sub_thickness*np.zeros_like(pattern)
        CNN_in = np.stack([pattern, sub_info], axis = 0)
        CNN_in = torch.tensor(CNN_in, dtype=torch.float32)
        
        S21C = self.log_normalize(group['S21C'][:self.tot_len_coarse]).reshape((self.step, self.len_coarse))
        S21D = self.log_normalize(group['S21D'][:self.tot_len_dense]).reshape((self.step, self.len_dense))
        
        LSTM_in = torch.tensor(S21C, dtype=torch.float32)
        HybridNN_out = torch.tensor(S21D, dtype=torch.float32)

        return CNN_in, LSTM_in, HybridNN_out

    def close(self):
        self.h5_file.close()
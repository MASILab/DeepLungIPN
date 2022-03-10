import torch.utils.data as data
from PIL import Image
import os
import os.path
import numpy as np
import random
import torch
import pandas as pd
import h5py
import pdb

class MultiPath_loader(data.Dataset):
    def __init__(self, list_IDs, dict_paths, path_diags, path_factors, path_labels, lastscan):
        self.path_labels = path_labels
        self.list_IDs = list_IDs
        self.dict_paths = dict_paths
        self.lastscan = lastscan
        self.dict_diags = path_diags
        self.path_factors = path_factors
        self.all_paths = self.get_all_IDs()

    def get_all_IDs(self):
            all_paths = []
            for ID in self.list_IDs:
                tmp_list = self.dict_paths[ID]
                for path in tmp_list:
                    all_paths.append(path)

            return all_paths
        
    def __len__(self):
            if self.lastscan:
                return len(self.list_IDs)
            else:
                return len(self.all_paths)

    def __getitem__(self, index):
            if self.lastscan == True:
                ID = self.list_IDs[index]
                path = sorted(self.dict_paths[ID])[-1]
            else:
                path = self.all_paths[index]
                
            factors = np.array(self.path_factors[path]).astype('float32')
            y = self.path_labels[path] 
            
            if 'default' in path:
                factors[0] = 0 # change at 0909
                img = np.zeros((5, 128)).astype('float32')
            else:
                img = np.load(path)
            
            if factors[1] == 0:
                factors[2:] = np.zeros(factors[2:].shape)
            
            return img, factors, y, path.split('/')[-1]


class MultiPath_loaderv2(data.Dataset):
    def __init__(self, list_IDs, path_factors, path_labels):
        self.path_labels = path_labels
        self.list_IDs = list_IDs

        self.path_factors = path_factors

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        
        path = self.list_IDs[index]
        
        factors = np.array(self.path_factors[path]).astype('float32')
        y = self.path_labels[path] 
        #print (factors.shape)
        if 'default' in path:
            factors[0] = 0 # change at 0909
            img = np.nan * np.zeros((5, 128)).astype('float32')
        else:
            img = np.load(path)

        if factors[1] == 0:
            factors[2:] = np.nan * np.zeros(factors[2:].shape)

        return img, factors, y, path.split('/')[-1]
        

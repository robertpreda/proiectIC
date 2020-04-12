import os
import torch

import pandas as pd
import numpy as np 

from torch.utils.data import Dataset
from torchvision import transforms, utils

def to_categorical(y, num_classes):
    return np.eye(num_classes)[y]

def to_img(data):
    converted_data = []
    for d in data:
        temp = int(d)
        converted_data.append(d)
    converted_data = np.array(converted_data).astype(np.uint8)
    converted_data = converted_data.reshape((48,48))
    return converted_data

class DataLoaderFacesICML2013(Dataset):
    def __init__(self, csv_file_path, root_dir, transform=None):
        self.faces_frame = pd.read_csv(csv_file_path)
        self.root_dir = root_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.faces_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img = to_img(self.faces_frame.iloc[idx, 1])
        label = to_categorical(self.faces_frame.iloc[idx, 0], 7)
        sample = {'image':img, 'label':label}

        if self.transform:
            sample = self.transform(sample)
        
        return sample

        
        

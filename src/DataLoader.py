import cv2
import os
import torch
import pandas as pd
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

label_dict ={
    'A':np.array([1,0,0,0]).astype(np.float),
    'B':np.array([0,1,0,0]).astype(np.float),
    'L':np.array([0,0,1,0]).astype(np.float),
    'W':np.array([0,0,0,1]).astype(np.float)
}

class EthnicityDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.frame = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = self.frame.iloc[idx, 0]
        image = cv2.imread(img_name)
        image = cv2.resize(image, (550, 550))
        label_letter = self.frame.iloc[idx, 1]
        label_vector = label_dict[label_letter]
        sample = {'image': image, 'label':torch.from_numpy(label_vector)}

        if self.transform:
            sample['image'] = self.transform(sample['image'])

        return sample
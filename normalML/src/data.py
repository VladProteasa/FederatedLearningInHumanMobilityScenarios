import os.path
import sys
import torch

from datetime import datetime
from torch.utils.data import Dataset
from sklearn.utils import shuffle


class Data(Dataset):
    def __init__(self, time, destination):
        self.users = {}
        seq = []
        # self.destination = destination
        # for key, (lats, lons, times) in users.items():
        #     for lat, lon, time in zip(lats, lons, times):
        #         seq.append((lat, lon, time))
        #     self.users[key] = torch.as_tensor(seq.copy())
            # seq.clear()

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.destination[idx]

    def shuffle_data(self):
        self.users, self.destination = shuffle(self.users, self.destination)
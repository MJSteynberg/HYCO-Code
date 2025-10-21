#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: MJ Steynberg
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

class DataLoader_Scalar(Dataset):
    """
        Load data 
    """
    def __init__(self, device, folder, flag):
        self.device = device
        filename = folder + f'/data_{flag}.npy'
        filename_t = folder + f'/t_{flag}.npy'
        if filename:
            u = np.load(filename)
        else:
            # Create dataset
            pass
        if os.path.exists(filename_t):
            t = np.load(filename_t)
        else:
            t = np.linspace(0, 1, u.shape[0])

        self.t = torch.tensor(t, dtype=torch.float32, device=self.device)
        self.u = torch.tensor(u, dtype=torch.float32, device=self.device)

    def __getitem__(self, index):
        return self.t[index], self.u[index, :, :]

    def length_t(self):
        return len(self.t)

    def length_u(self):
        return self.u.shape[1]
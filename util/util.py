import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np

def load_data_percentage(path_X, path_Y, path_Z, percentage=10):
    """
    Load a percentage of the data from the specified paths and normalize labels.
    
    :param path_X: str - Path to the X.npy file
    :param path_Y: str - Path to the Y.npy file
    :param path_Z: str - Path to the Z.npy file
    :param percentage: int - Percentage of data to load (0 < percentage <= 100)
    :return: tuple - (X, Y, Z) arrays with the specified percentage of data
    """
    # Load the full data
    X = np.load(path_X)
    Y = np.load(path_Y)
    Z = np.load(path_Z)
    
    # Calculate the number of samples to load
    num_samples = int(len(X) * (percentage / 100))
    
    # Slice the data to get the required percentage
    X = X[:num_samples]
    Y = Y[:num_samples]
    Z = Z[:num_samples]


    return X, Y, Z


class CustomDataset(Dataset):
    def __init__(self, X, Y, Z):
        self.X = torch.tensor(X, dtype=torch.float32).permute(0, 2, 1)
        self.Y = torch.tensor(Y, dtype=torch.float32)
        self.Z = torch.tensor(Z, dtype=torch.float32).squeeze(-2)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.Z[idx]


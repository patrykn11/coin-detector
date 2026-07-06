import pickle
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor


class CoinDetectionDataset(Dataset):
    def __init__(self, path):
        with open(Path(path), "rb") as f:
            self.data = pickle.load(f)

    def __len__(self):
        return len(self.data)
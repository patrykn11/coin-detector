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

    def __getitem__(self, idx):
        sample = self.data[idx]

        return {
            "image": to_tensor(sample["image"].convert("RGB")),
            "boxes": torch.as_tensor(
                sample["boxes"], dtype=torch.float32
            ).reshape(-1, 4),
            "labels": torch.as_tensor(
                sample["labels"], dtype=torch.long
            ).reshape(-1),
        }
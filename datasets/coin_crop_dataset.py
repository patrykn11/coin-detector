import torch
from torch.utils.data import Dataset


class CoinCropDataset(Dataset):
    def __init__(self, crops, labels, bb_diffs):
        self.crops = torch.stack(crops)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.bb_diffs = torch.stack(bb_diffs)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.crops[idx],
            self.labels[idx],
            self.bb_diffs[idx],
        )
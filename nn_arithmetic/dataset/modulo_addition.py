import torch
from torch.utils.data import Dataset


class ModuloAdditionDataset(Dataset):
    def __init__(self, modulo_factor: int, train: bool = True, train_split: float = 0.8):
        self.modulo_factor = modulo_factor
        self.train = train

        all_pairs = []
        for a in range(modulo_factor):
            for b in range(modulo_factor):
                result = (a + b) % modulo_factor
                all_pairs.append((a, b, result))

        split_idx = int(len(all_pairs) * train_split)
        self.data = all_pairs[:split_idx] if train else all_pairs[split_idx:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        a, b, result = self.data[idx]
        return torch.tensor([a, b], dtype=torch.long), torch.tensor(result, dtype=torch.long)

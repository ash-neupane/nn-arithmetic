from pathlib import Path

import torch
from torch.utils.data import Dataset

from nn_arithmetic.tokenizer import ShakespeareTokenizer


class TinyShakespeareDataset(Dataset):
    def __init__(
        self,
        data_dir: str | None = None,
        seq_length: int = 128,
        train: bool = True,
    ):
        if data_dir is None:
            data_dir = Path(__file__).parent.parent.parent / "data" / "tiny_shakespeare"
        else:
            data_dir = Path(data_dir)

        token_file = data_dir / ("train.pt" if train else "val.pt")

        if not token_file.exists():
            raise FileNotFoundError(
                f"Pre-tokenized data not found at {token_file}. "
                "Please run scripts/tokenize_shakespeare.sh first to tokenize the dataset."
            )

        self.seq_length = seq_length
        self.tokens = torch.load(token_file)
        self.tokenizer = ShakespeareTokenizer()

    def __len__(self):
        return max(0, len(self.tokens) - self.seq_length)

    def __getitem__(self, idx):
        chunk = self.tokens[idx : idx + self.seq_length + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size

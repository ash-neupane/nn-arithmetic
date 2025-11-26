from pathlib import Path

import torch
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer


class ShakespeareTokenizer:
    """Tokenizer for tiny-shakespeare dataset."""

    def __init__(self, tokenizer_path: str | None = None):
        """
        Load tokenizer from file.

        Args:
            tokenizer_path: Path to tokenizer JSON file. Defaults to artifacts/tiny_shakespeare_tokenizer.json
        """
        if tokenizer_path is None:
            tokenizer_path = Path(__file__).parent.parent.parent / "artifacts" / "tiny_shakespeare_tokenizer.json"
        else:
            tokenizer_path = Path(tokenizer_path)

        if not tokenizer_path.exists():
            raise FileNotFoundError(
                f"Tokenizer not found at {tokenizer_path}. "
                "Please run 'python -m nn_arithmetic.tokenizer.shakespeare' first to train the tokenizer."
            )

        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs."""
        return self.tokenizer.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs to text."""
        return self.tokenizer.decode(ids)

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return self.tokenizer.get_vocab_size()

    @staticmethod
    def train(
        data_path: str | None = None,
        output_path: str | None = None,
        vocab_size: int = 1000,
    ) -> "ShakespeareTokenizer":
        """
        Train a BPE tokenizer on tiny-shakespeare dataset and save it.

        Args:
            data_path: Path to input.txt (defaults to data/tiny_shakespeare/input.txt)
            output_path: Path to save tokenizer (defaults to artifacts/tiny_shakespeare_tokenizer.json)
            vocab_size: Vocabulary size for BPE

        Returns:
            Trained tokenizer instance
        """
        if data_path is None:
            data_path = Path(__file__).parent.parent.parent / "data" / "tiny_shakespeare" / "input.txt"
        else:
            data_path = Path(data_path)

        if output_path is None:
            output_path = Path(__file__).parent.parent.parent / "artifacts" / "tiny_shakespeare_tokenizer.json"
        else:
            output_path = Path(output_path)

        if not data_path.exists():
            raise FileNotFoundError(
                f"Data file not found at {data_path}. "
                "Please run scripts/download_tiny_shakespeare.sh first to download the dataset."
            )

        print(f"Training BPE tokenizer on {data_path}...")

        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()

        trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=["[UNK]", "[PAD]"])
        tokenizer.train([str(data_path)], trainer)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        tokenizer.save(str(output_path))

        print(f"Tokenizer trained successfully!")
        print(f"  Vocabulary size: {tokenizer.get_vocab_size()}")
        print(f"  Saved to: {output_path}")

        return ShakespeareTokenizer(str(output_path))

    @staticmethod
    def tokenize_and_save(
        data_path: str | None = None,
        tokenizer_path: str | None = None,
        output_dir: str | None = None,
        train_split: float = 0.9,
    ):
        """
        Tokenize the dataset and save train/val splits as .pt files.

        Args:
            data_path: Path to input.txt
            tokenizer_path: Path to tokenizer JSON file
            output_dir: Directory to save train.pt and val.pt
            train_split: Fraction of data to use for training
        """
        if data_path is None:
            data_path = Path(__file__).parent.parent.parent / "data" / "tiny_shakespeare" / "input.txt"
        else:
            data_path = Path(data_path)

        if output_dir is None:
            output_dir = Path(__file__).parent.parent.parent / "data" / "tiny_shakespeare"
        else:
            output_dir = Path(output_dir)

        print(f"Loading text from {data_path}...")
        with open(data_path, "r", encoding="utf-8") as f:
            text = f.read()

        print("Tokenizing...")
        tokenizer = ShakespeareTokenizer(tokenizer_path)
        tokens = tokenizer.encode(text)

        split_idx = int(len(tokens) * train_split)
        train_tokens = tokens[:split_idx]
        val_tokens = tokens[split_idx:]

        output_dir.mkdir(parents=True, exist_ok=True)

        train_path = output_dir / "train.pt"
        val_path = output_dir / "val.pt"

        print(f"Saving tokenized data...")
        torch.save(torch.tensor(train_tokens, dtype=torch.long), train_path)
        torch.save(torch.tensor(val_tokens, dtype=torch.long), val_path)

        print(f"Tokenization complete!")
        print(f"  Train tokens: {len(train_tokens):,} -> {train_path}")
        print(f"  Val tokens: {len(val_tokens):,} -> {val_path}")


if __name__ == "__main__":
    print("Training tokenizer...")
    ShakespeareTokenizer.train()

    print("\nTokenizing dataset...")
    ShakespeareTokenizer.tokenize_and_save()

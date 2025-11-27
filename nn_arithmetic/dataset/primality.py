import random

import numpy as np
import torch
from torch.utils.data import Dataset


def is_prime(n: int) -> bool:
    """Check if n is prime using trial division."""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True


def miller_rabin(n: int, k: int = 5) -> bool:
    """Miller-Rabin primality test for large numbers."""
    if n < 2:
        return False
    if n == 2 or n == 3:
        return True
    if n % 2 == 0:
        return False

    # Write n-1 as 2^r * d
    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2

    # Witness loop
    for _ in range(k):
        a = random.randrange(2, n - 1)
        x = pow(a, d, n)

        if x == 1 or x == n - 1:
            continue

        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False

    return True


def number_to_binary_tensor(n: int, num_bits: int = 64) -> torch.Tensor:
    """Convert integer to binary tensor representation."""
    binary_str = format(n, f"0{num_bits}b")
    bits = [int(b) for b in binary_str]
    return torch.tensor(bits, dtype=torch.float32)


class PrimalityDataset(Dataset):
    """Dataset for primality testing up to 2^64."""

    def __init__(
        self,
        num_samples: int = 100000,
        max_bits: int = 16,
        train: bool = True,
        train_split: float = 0.8,
        seed: int = 42,
        balanced: bool = True,
    ):
        """
        Args:
            num_samples: Number of samples to generate
            max_bits: Maximum number of bits for numbers (1 to 64)
            train: Whether this is training or validation set
            train_split: Fraction of data for training
            seed: Random seed for reproducibility
            balanced: If True, balance primes and composites
        """
        self.max_bits = max_bits
        self.num_bits = 64  # Always use 64-bit representation
        random.seed(seed)
        np.random.seed(seed)

        # Generate samples
        self.samples = []
        max_val = 2**max_bits

        if balanced:
            # Generate equal numbers of primes and composites
            primes_needed = num_samples // 2
            composites_needed = num_samples - primes_needed

            # Generate primes
            primes = []
            attempts = 0
            max_attempts = primes_needed * 100

            while len(primes) < primes_needed and attempts < max_attempts:
                n = random.randint(2, max_val - 1)
                if max_bits <= 20:
                    if is_prime(n):
                        primes.append((n, 1))
                else:
                    if miller_rabin(n, k=10):
                        primes.append((n, 1))
                attempts += 1

            # Generate composites
            composites = []
            while len(composites) < composites_needed:
                n = random.randint(2, max_val - 1)
                if max_bits <= 20:
                    if not is_prime(n):
                        composites.append((n, 0))
                else:
                    if not miller_rabin(n, k=10):
                        composites.append((n, 0))

            self.samples = primes + composites
        else:
            # Random sampling
            for _ in range(num_samples):
                n = random.randint(2, max_val - 1)
                if max_bits <= 20:
                    label = 1 if is_prime(n) else 0
                else:
                    label = 1 if miller_rabin(n, k=10) else 0
                self.samples.append((n, label))

        # Shuffle
        random.shuffle(self.samples)

        # Split train/val
        split_idx = int(len(self.samples) * train_split)
        if train:
            self.samples = self.samples[:split_idx]
        else:
            self.samples = self.samples[split_idx:]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        n, label = self.samples[idx]
        x = number_to_binary_tensor(n, self.num_bits)
        y = torch.tensor(label, dtype=torch.long)
        return x, y

    def get_prime_ratio(self):
        """Get the ratio of primes in the dataset."""
        primes = sum(1 for _, label in self.samples if label == 1)
        return primes / len(self.samples)

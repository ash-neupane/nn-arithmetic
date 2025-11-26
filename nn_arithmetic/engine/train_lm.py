import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from nn_arithmetic.dataset.tiny_shakespeare import TinyShakespeareDataset
from nn_arithmetic.engine.logging import WBLogger
from nn_arithmetic.model.language_model import GPTTransformer
from nn_arithmetic.tokenizer import ShakespeareTokenizer


def train_language_model(epochs: int = 100, batch_size: int = 16, seq_length: int = 128, lr: float = 3e-4):
    d_model = 128
    nhead = 4
    num_layers = 4
    dim_feedforward = 512

    train_dataset = TinyShakespeareDataset(seq_length=seq_length, train=True)
    val_dataset = TinyShakespeareDataset(seq_length=seq_length, train=False)

    config = {
        "epochs": epochs,
        "batch_size": batch_size,
        "seq_length": seq_length,
        "d_model": d_model,
        "nhead": nhead,
        "num_layers": num_layers,
        "dim_feedforward": dim_feedforward,
        "lr": lr,
        "vocab_size": train_dataset.vocab_size,
    }

    logger = WBLogger(project="nn-arithmetic-lm", config=config)
    device = torch.device("cpu")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = GPTTransformer(
        vocab_size=train_dataset.vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        max_seq_length=seq_length,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    total_steps = epochs * len(train_loader)
    pbar = tqdm(total=total_steps, desc="Training")
    global_step = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            global_step += 1
            logger.log({"train/loss": loss.item()}, step=global_step)

            pbar.update(1)
            pbar.set_postfix({"epoch": epoch + 1, "loss": f"{loss.item():.4f}"})

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                logits = model(inputs)
                loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        logger.log(
            {
                "epoch": epoch + 1,
                "train/avg_loss": avg_train_loss,
                "val/avg_loss": avg_val_loss,
            },
            step=global_step,
        )

        if (epoch + 1) % 10 == 0:
            pbar.write(
                f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}"
            )

    pbar.close()
    logger.finish()

    tokenizer = ShakespeareTokenizer()
    return model, tokenizer

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from nn_arithmetic.dataset.primality import PrimalityDataset
from nn_arithmetic.engine.logging import WBLogger
from nn_arithmetic.model.primality_classifier import PrimalityMLP, PrimalityTransformer


def train_primality_classifier(
    max_bits: int = 16,
    num_samples: int = 100000,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    model_type: str = "transformer",
    use_curriculum: bool = False,
):
    """
    Train a primality classifier.

    Args:
        max_bits: Maximum number of bits for training numbers (1-64)
        num_samples: Number of samples to generate
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        model_type: "transformer" or "mlp"
        use_curriculum: If True, use curriculum learning (start small, increase bit size)
    """
    # Model hyperparameters
    if model_type == "transformer":
        d_model = 128
        nhead = 4
        num_layers = 4
        dim_feedforward = 512
    else:
        hidden_dim = 512
        num_layers = 3

    # Create datasets
    train_dataset = PrimalityDataset(
        num_samples=num_samples,
        max_bits=max_bits,
        train=True,
        balanced=True,
    )
    val_dataset = PrimalityDataset(
        num_samples=num_samples // 5,
        max_bits=max_bits,
        train=False,
        balanced=True,
    )

    config = {
        "max_bits": max_bits,
        "num_samples": num_samples,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "model_type": model_type,
        "train_prime_ratio": train_dataset.get_prime_ratio(),
        "val_prime_ratio": val_dataset.get_prime_ratio(),
    }

    if model_type == "transformer":
        config.update({
            "d_model": d_model,
            "nhead": nhead,
            "num_layers": num_layers,
            "dim_feedforward": dim_feedforward,
        })
    else:
        config.update({
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
        })

    logger = WBLogger(project="nn-arithmetic-primality", config=config)
    device = torch.device("cpu")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Create model
    if model_type == "transformer":
        model = PrimalityTransformer(
            num_bits=64,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
        ).to(device)
    else:
        model = PrimalityMLP(
            num_bits=64,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    total_steps = epochs * len(train_loader)
    pbar = tqdm(total=total_steps, desc="Training")
    global_step = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = logits.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()

            global_step += 1
            logger.log({"train/loss": loss.item()}, step=global_step)

            pbar.update(1)
            pbar.set_postfix({"epoch": epoch + 1, "loss": f"{loss.item():.4f}"})

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_true_positives = 0
        val_false_positives = 0
        val_true_negatives = 0
        val_false_negatives = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                logits = model(inputs)
                loss = criterion(logits, targets)
                val_loss += loss.item()

                _, predicted = logits.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()

                # Confusion matrix
                val_true_positives += ((predicted == 1) & (targets == 1)).sum().item()
                val_false_positives += ((predicted == 1) & (targets == 0)).sum().item()
                val_true_negatives += ((predicted == 0) & (targets == 0)).sum().item()
                val_false_negatives += ((predicted == 0) & (targets == 1)).sum().item()

        train_acc = 100.0 * train_correct / train_total
        val_acc = 100.0 * val_correct / val_total
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        # Precision and recall
        precision = 100.0 * val_true_positives / max(1, val_true_positives + val_false_positives)
        recall = 100.0 * val_true_positives / max(1, val_true_positives + val_false_negatives)
        f1 = 2 * precision * recall / max(1, precision + recall)

        logger.log(
            {
                "epoch": epoch + 1,
                "train/avg_loss": avg_train_loss,
                "train/accuracy": train_acc,
                "val/avg_loss": avg_val_loss,
                "val/accuracy": val_acc,
                "val/precision": precision,
                "val/recall": recall,
                "val/f1": f1,
            },
            step=global_step,
        )

        if (epoch + 1) % 10 == 0:
            pbar.write(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
                f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | "
                f"F1: {f1:.2f}%"
            )

    pbar.close()
    logger.finish()

    return model


if __name__ == "__main__":
    model = train_primality_classifier(max_bits=16, epochs=200)

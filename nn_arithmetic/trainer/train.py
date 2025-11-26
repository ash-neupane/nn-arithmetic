import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from nn_arithmetic.dataset.modulo_addition import ModuloAdditionDataset
from nn_arithmetic.model.transformer import ModuloTransformer
from nn_arithmetic.trainer.logging import WBLogger


def train_modulo_adder(modulo_factor: int = 113, epochs: int = 100, batch_size: int = 64):
    d_model = 64
    nhead = 4
    num_layers = 2
    lr = 1e-3

    config = {
        "modulo_factor": modulo_factor,
        "epochs": epochs,
        "batch_size": batch_size,
        "d_model": d_model,
        "nhead": nhead,
        "num_layers": num_layers,
        "lr": lr,
    }

    logger = WBLogger(project="nn-arithmetic", config=config)
    device = torch.device("cpu")

    train_dataset = ModuloAdditionDataset(modulo_factor, train=True)
    val_dataset = ModuloAdditionDataset(modulo_factor, train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = ModuloTransformer(modulo_factor=modulo_factor, d_model=d_model, nhead=nhead, num_layers=num_layers).to(
        device
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

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
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()

            global_step += 1
            logger.log({"train/loss": loss.item()}, step=global_step)

            pbar.update(1)
            pbar.set_postfix({"epoch": epoch + 1, "loss": f"{loss.item():.4f}"})

        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()

        train_acc = 100.0 * train_correct / train_total
        val_acc = 100.0 * val_correct / val_total
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        logger.log(
            {
                "epoch": epoch + 1,
                "train/avg_loss": avg_train_loss,
                "train/accuracy": train_acc,
                "val/avg_loss": avg_val_loss,
                "val/accuracy": val_acc,
            },
            step=global_step,
        )

        if (epoch + 1) % 10 == 0:
            pbar.write(
                f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
                f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%"
            )

    pbar.close()
    logger.finish()

    return model

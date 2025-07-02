# train.py

import os
import yaml
import argparse
from typing import Any, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.data_utils import create_dataloaders
from src.model_builder import build_model


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
) -> float:
    """Performs one full training epoch."""
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(
        enumerate(train_loader),
        desc=f"Epoch {epoch+1} [Training]",
    )

    for i, batch in progress_bar:
        videos = [video.to(device) for video in batch["video"]]
        labels = batch["label"].to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        preds = model(videos)

        # Compute loss
        loss = loss_fn(preds, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Log training loss for each batch
        writer.add_scalar("Loss/train_batch", loss.item(), epoch * 1000 + i)  # Use epoch * 1000 as approximation
        progress_bar.set_postfix({"loss": loss.item()})
    
    avg_loss = total_loss / max(1, i + 1)  # Use actual number of batches processed
    return avg_loss


def validate_one_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Performs one full validation epoch."""
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    batch_count = 0

    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc=f"Epoch [Validation]")
        for batch in progress_bar:
            batch_count += 1
            videos = [video.to(device) for video in batch["video"]]
            labels = batch["label"].to(device)

            # Forward pass
            preds = model(videos)
            loss = loss_fn(preds, labels)
            total_loss += loss.item()

            # Calculate accuracy
            _, predicted_labels = torch.max(preds, 1)
            correct_predictions += (predicted_labels == labels).sum().item()
            total_samples += labels.size(0)

            accuracy = correct_predictions / total_samples
            progress_bar.set_postfix({"accuracy": f"{accuracy:.4f}"})

    avg_loss = total_loss / max(1, batch_count)
    final_accuracy = correct_predictions / total_samples
    return avg_loss, final_accuracy


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_accuracy: float,
    train_loss: float,
    val_loss: float,
    val_accuracy: float,
    checkpoint_path: str
) -> None:
    """Save complete training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_accuracy': best_val_accuracy,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'val_accuracy': val_accuracy,
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")


def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: str,
    device: torch.device
) -> tuple[int, float]:
    """Load training checkpoint and return start epoch and best accuracy."""
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        start_epoch = checkpoint['epoch'] + 1  # Start from next epoch
        best_val_accuracy = checkpoint['best_val_accuracy']
        
        print(f"Resuming training from epoch {start_epoch}")
        print(f"Previous best validation accuracy: {best_val_accuracy:.4f}")
        
        return start_epoch, best_val_accuracy
    else:
        print("No checkpoint found, starting training from scratch")
        return 0, 0.0


def main(config_path: str) -> None:
    """Main function to run the training pipeline."""
    # --- 1. Load Configuration ---
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # --- 2. Setup Device, Logging, and Checkpointing ---
    device = torch.device(config["training"]["device"] if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(log_dir=config["logging"]["log_dir"])
    os.makedirs(config["logging"]["checkpoint_dir"], exist_ok=True)
    
    # Define checkpoint paths
    resume_checkpoint_path = os.path.join(
        config["logging"]["checkpoint_dir"], "latest_checkpoint.pth"
    )
    best_model_path = os.path.join(
        config["logging"]["checkpoint_dir"], config["logging"]["checkpoint_filename"]
    )

    # --- 3. Create DataLoaders and Model ---
    train_loader, val_loader = create_dataloaders(config)
    model = build_model(config, device)

    # --- 4. Define Loss Function and Optimizer ---
    loss_fn = nn.CrossEntropyLoss()
    optimizer_name = config["training"]["optimizer"].lower()
    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config["training"]["learning_rate"], weight_decay=config["training"]["weight_decay"]
        )
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=config["training"]["learning_rate"], momentum=0.9, weight_decay=config["training"]["weight_decay"]
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    # --- 5. Load Checkpoint if Available ---
    start_epoch, best_val_accuracy = load_checkpoint(model, optimizer, resume_checkpoint_path, device)

    # --- 6. Training Loop ---
    print("\n--- Starting Training ---")
    for epoch in range(start_epoch, config["training"]["epochs"]):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, epoch, writer
        )
        writer.add_scalar("Loss/train_epoch", train_loss, epoch)
        print(f"\nEpoch {epoch+1} | Average Training Loss: {train_loss:.4f}")

        # Validate
        val_loss, val_accuracy = validate_one_epoch(model, val_loader, loss_fn, device)
        writer.add_scalar("Loss/val_epoch", val_loss, epoch)
        writer.add_scalar("Accuracy/val", val_accuracy, epoch)
        print(f"Epoch {epoch+1} | Validation Loss: {val_loss:.4f} | Validation Accuracy: {val_accuracy:.4f}")

        # Save checkpoint after each epoch (for resuming)
        save_checkpoint(
            model, optimizer, epoch, best_val_accuracy, 
            train_loss, val_loss, val_accuracy, resume_checkpoint_path
        )

        # Save the best model separately
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), best_model_path)
            print(f"-> New best model saved to {best_model_path} with accuracy: {val_accuracy:.4f}")

    writer.close()
    print("\n--- Training Finished ---")
    print(f"Best validation accuracy: {best_val_accuracy:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a video classification model.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to the configuration YAML file.",
    )
    args = parser.parse_args()
    main(args.config) 
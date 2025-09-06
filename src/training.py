# Imports

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm
from typing import List

# Local imports

from src.dataset import IterablePositionsDataset
from src.early_stopping import EarlyStopping


def create_dataloaders(
    parquet_paths: List[str],
    batch_size: int,
    train_percent: float = 0.8,
    shuffle_buffer_size: int = 100_000,
) -> tuple[DataLoader, DataLoader]:
    """
    Creates training and validation dataloaders from an iterable dataset.
    Splits the list of parquet files into training and validation sets.
    """
    num_files = len(parquet_paths)

    if num_files < 2:
        print("Warning: Only one Parquet file provided. Using it for both training and validation.")
        train_files = parquet_paths
        val_files = parquet_paths
    else:
        split_index = int(num_files * train_percent)
        # Ensure at least one file for validation
        if split_index == num_files:
            split_index -= 1
        
        train_files = parquet_paths[:split_index]
        val_files = parquet_paths[split_index:]

    print(f"Using {len(train_files)} files for training and {len(val_files)} for validation.")

    # Create a dataset instance for the training split with shuffling
    train_dataset = IterablePositionsDataset(
        train_files, shuffle_buffer_size=shuffle_buffer_size
    )

    # Create a dataset instance for the validation split without shuffling
    val_dataset = IterablePositionsDataset(
        val_files, shuffle_buffer_size=0  # No need to shuffle validation data
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, num_workers=10, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, num_workers=10, pin_memory=True
    )

    return train_loader, val_loader


def validation(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: list[nn.Module],
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation Progress"):
            inputs = batch["board_tensor"].to(device)
            best_move_target = batch["best_move"].to(device)
            value_target = batch["value_target"].to(device)
            game_state_target = batch["game_state_target"].to(device)

            with autocast(device_type=device.type):
                outputs = model(inputs)

                best_move_loss = criterion[0](outputs["best_move"], best_move_target)
                value_loss = criterion[1](outputs["value"].squeeze(), value_target)
                game_state_loss = criterion[2](outputs["game_state"], game_state_target)

                total_loss += (best_move_loss + value_loss + game_state_loss).item()

    avg_loss = total_loss / len(val_loader)
    return avg_loss


def training(
    model: nn.Module,
    epochs: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: ReduceLROnPlateau,
    early_stopping: EarlyStopping,
    criterion: list[nn.Module],
    save_dir: str,
    device: torch.device,
):
    # We will store the losses
    training_loss = []
    validation_loss = []
    best_loss = float("inf")

    scaler = GradScaler()

    for epoch in range(epochs):

        print(f"Starting epoch {epoch+1}/{epochs}")

        total_train_loss = 0.0

        # Start training loop
        model.train()

        for i, batch in tqdm(
            enumerate(train_loader), desc="Training Progress", total=len(train_loader)
        ):

            # We get the inputs and targets
            inputs = batch["board_tensor"].to(device)
            best_move_target = batch["best_move"].to(device)
            value_target = batch["value_target"].to(device)
            game_state_target = batch["game_state_target"].to(device)

            with autocast(device_type=device.type):
                # Get the model outputs
                outputs = model(inputs)

                # Compute the losses
                best_move_loss = criterion[0](outputs["best_move"], best_move_target)
                value_loss = criterion[1](outputs["value"].squeeze(), value_target)
                game_state_loss = criterion[2](outputs["game_state"], game_state_target)

                total_loss = best_move_loss + value_loss + game_state_loss

            # Backpropagation
            optimizer.zero_grad()
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_train_loss += total_loss.item()

        # Compute the average training loss
        avg_train_loss = total_train_loss / len(train_loader)

        # Validation step
        avg_val_loss = validation(model, val_loader, criterion, device)

        # Log the losses
        training_loss.append(avg_train_loss)
        validation_loss.append(avg_val_loss)

        print(
            f"Epoch {epoch+1}/{epochs}, Training Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}"
        )

        # Step the scheduler
        scheduler.step(avg_val_loss)

        # Check early stopping
        if early_stopping(avg_val_loss):
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

        # Save the model if it is our best loss
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), f"{save_dir}/best_model.pth")
            print(f"Best model saved with validation loss: {best_loss}")

            # Save the optimizer state
            optimizer_state = optimizer.state_dict()
            torch.save(optimizer_state, f"{save_dir}/best_optimizer.pth")

            # Save the scheduler state
            scheduler_state = scheduler.state_dict()
            torch.save(scheduler_state, f"{save_dir}/best_scheduler.pth")

    return training_loss, validation_loss
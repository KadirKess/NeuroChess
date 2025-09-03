# Imports

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Local imports

from torch.utils.data import DataLoader
from src.dataset import IterablePositionsDataset  # Updated import


def create_dataloaders(
    parquet_path: str, batch_size: int, train_percent: float = 0.8
) -> tuple[DataLoader, DataLoader]:
    """
    Creates training and validation dataloaders from an iterable dataset.
    Note: Iterable datasets can't be split randomly after creation. We create
    two separate dataset instances that stream from different portions of the file.
    """
    # Create a dataset instance for the training split
    train_dataset = IterablePositionsDataset(
        parquet_path, start_frac=0.0, end_frac=train_percent
    )

    # Create a dataset instance for the validation split
    val_dataset = IterablePositionsDataset(
        parquet_path, start_frac=train_percent, end_frac=1.0
    )

    # Note: For iterable datasets, 'shuffle' is not applicable in the same way.
    # Shuffling is often handled by the dataset's __iter__ method if needed.
    # Here, we rely on the inherent randomness of the source data.
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, num_workers=2, pin_memory=True
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
    criterion: list[nn.Module],
    save_dir: str,
    device: torch.device,
):
    # We will store the losses
    training_loss = []
    validation_loss = []
    best_loss = float("inf")

    for epoch in range(epochs):

        print(f"Starting epoch {epoch+1}/{epochs}")

        total_train_loss = 0.0

        # Start training loop
        model.train()

        for i, batch in tqdm(enumerate(train_loader), desc="Training Progress"):

            # We get the inputs and targets
            inputs = batch["board_tensor"].to(device)
            best_move_target = batch["best_move"].to(device)
            value_target = batch["value_target"].to(device)
            game_state_target = batch["game_state_target"].to(device)

            # Get the model outputs
            outputs = model(inputs)

            # Compute the losses
            best_move_loss = criterion[0](outputs["best_move"], best_move_target)
            value_loss = criterion[1](outputs["value"].squeeze(), value_target)
            game_state_loss = criterion[2](outputs["game_state"], game_state_target)

            total_loss = best_move_loss + value_loss + game_state_loss

            # Backpropagation
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            total_train_loss += total_loss.item()

            # Once ever 200 iterations, print the training loss
            if i % 200 == 0:
                print(
                    f"Epoch {epoch+1}, Iteration {i}, Training Loss: {total_loss.item()}"
                )

        # Compute the average training loss
        avg_train_loss = total_train_loss / len(train_loader)

        # Validation step
        avg_val_loss = validation(model, val_loader, criterion, device)

        # Log the losses
        training_loss.append(avg_train_loss)
        validation_loss.append(avg_val_loss)

        print(
            f"Epoch {epoch+1}/{epochs}, Training Loss: {avg_train_loss.item()}, Validation Loss: {avg_val_loss}"
        )

        # Step the scheduler
        scheduler.step(avg_val_loss)

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

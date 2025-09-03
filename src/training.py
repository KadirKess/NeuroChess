# Imports

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


def create_dataloaders(
    dataset: Dataset, batch_size: int, shuffle: bool = True, train_percent: float = 0.8
) -> tuple[DataLoader, DataLoader]:
    train_size = int(train_percent * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=1
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=1
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

        for i, batch in tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Training progress",
        ):

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

        # Compute the average training loss
        avg_train_loss = total_train_loss / len(train_loader)

        # Validation step
        avg_val_loss = validation(model, val_loader, criterion, device)

        # Log the losses
        training_loss.append(avg_train_loss.item())
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

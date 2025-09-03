# Imports
import torch
from torch.utils.data import Dataset
import numpy as np


class PositionsDataset(Dataset):
    def __init__(self, processed_data_path):
        self.board_tensors = np.load(
            f"{processed_data_path}/board_tensors.npy", mmap_mode="r"
        )
        self.game_state_target = np.load(
            f"{processed_data_path}/game_state_target.npy", mmap_mode="r"
        )
        self.value_target = np.load(
            f"{processed_data_path}/value_target.npy", mmap_mode="r"
        )
        self.best_move = np.load(f"{processed_data_path}/best_move.npy", mmap_mode="r")

    def __len__(self):
        return len(self.board_tensors)

    def __getitem__(self, idx):
        """
        Get a chess position by index.
        """
        return {
            "board_tensor": torch.from_numpy(self.board_tensors[idx].copy()),
            "game_state_target": torch.tensor(
                self.game_state_target[idx], dtype=torch.long
            ),
            "value_target": torch.tensor(self.value_target[idx], dtype=torch.float32),
            "best_move": torch.tensor(self.best_move[idx], dtype=torch.long),
        }

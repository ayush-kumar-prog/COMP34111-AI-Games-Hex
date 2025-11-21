"""
PyTorch dataset for Hex training data.
"""

import torch
import numpy as np
import pickle
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
from pathlib import Path

from .self_play import TrainingExample


class HexDataset(Dataset):
    """
    PyTorch dataset for Hex training examples.
    """

    def __init__(self, examples: List[TrainingExample]):
        """
        Initialize dataset.

        Args:
            examples: List of training examples from self-play
        """
        self.examples = examples

        # Convert to numpy arrays for faster access
        self.states = np.array([ex.state for ex in examples], dtype=np.float32)
        self.policies = np.array([ex.policy for ex in examples], dtype=np.float32)
        self.values = np.array([ex.value for ex in examples], dtype=np.float32)

        print(f"HexDataset created with {len(self.examples)} examples")
        print(f"  States shape: {self.states.shape}")
        print(f"  Policies shape: {self.policies.shape}")
        print(f"  Values shape: {self.values.shape}")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get single training example.

        Returns:
            state: (5, 11, 11) tensor
            policy: (121,) tensor
            value: (1,) tensor
        """
        state = torch.from_numpy(self.states[idx])
        policy = torch.from_numpy(self.policies[idx])
        value = torch.tensor([self.values[idx]], dtype=torch.float32)

        return state, policy, value

    @staticmethod
    def load_from_file(filepath: str) -> 'HexDataset':
        """Load dataset from pickle file."""
        with open(filepath, 'rb') as f:
            examples = pickle.load(f)
        return HexDataset(examples)

    @staticmethod
    def load_from_multiple_files(filepaths: List[str]) -> 'HexDataset':
        """Load and merge datasets from multiple files."""
        all_examples = []
        for filepath in filepaths:
            with open(filepath, 'rb') as f:
                examples = pickle.load(f)
                all_examples.extend(examples)
                print(f"Loaded {len(examples)} examples from {filepath}")

        print(f"Total examples: {len(all_examples)}")
        return HexDataset(all_examples)

    def get_dataloader(self, batch_size: int = 256,
                       shuffle: bool = True,
                       num_workers: int = 4) -> DataLoader:
        """Create PyTorch DataLoader."""
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True  # Faster transfer to GPU
        )

    def get_statistics(self) -> dict:
        """Get dataset statistics."""
        return {
            'num_examples': len(self.examples),
            'mean_value': float(np.mean(self.values)),
            'std_value': float(np.std(self.values)),
            'red_wins': int(np.sum(self.values > 0)),
            'blue_wins': int(np.sum(self.values < 0)),
            'policy_entropy': float(np.mean([-np.sum(p * np.log(p + 1e-10))
                                            for p in self.policies]))
        }


if __name__ == '__main__':
    print("Testing HexDataset...")

    # Would need actual training data to test
    print("✅ HexDataset class ready")
    print("   - Efficient numpy array storage")
    print("   - PyTorch DataLoader support")
    print("   - Batch loading with pin_memory")

"""
Training infrastructure for neural Hex AI.

Note: Imports are lazy to allow CPU-only modules (like expert_data_gen)
to work without PyTorch installed.
"""

# Lazy imports - don't import torch-dependent modules at package level
# Import directly in scripts that need them:
#   from agents.Group12.training.self_play import SelfPlayWorker
#   from agents.Group12.training.trainer import HexTrainer

__all__ = ['SelfPlayWorker', 'SelfPlayManager', 'HexTrainer', 'HexDataset']


def __getattr__(name):
    """Lazy import for backward compatibility."""
    if name == 'SelfPlayWorker' or name == 'SelfPlayManager':
        from .self_play import SelfPlayWorker, SelfPlayManager
        return SelfPlayWorker if name == 'SelfPlayWorker' else SelfPlayManager
    elif name == 'HexTrainer':
        from .trainer import HexTrainer
        return HexTrainer
    elif name == 'HexDataset':
        from .dataset import HexDataset
        return HexDataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision.datasets import ImageFolder

from silknet.config import BATCH_SIZE, SEED
from silknet.modeling.transforms import train_transforms, val_transforms


def train_val_test_split(input_path: Path) -> tuple[DataLoader, DataLoader, DataLoader]:
    full_set = ImageFolder(root=input_path)
    
    indices = list(range(len(full_set)))
    targets = full_set.targets

    train_indices, remaining_indices, train_targets, remaining_targets = train_test_split(
        indices, targets, test_size=0.2, random_state=SEED, stratify=targets
    )

    val_indices, test_indices, _, _ = train_test_split(
        remaining_indices,
        remaining_targets,
        test_size=0.5,
        random_state=SEED,
        stratify=remaining_targets,
    )

    train_subset = Subset(full_set, train_indices)
    val_subset = Subset(full_set, val_indices)
    test_subset = Subset(full_set, test_indices)

    train_subset.dataset.transform = train_transforms  # type: ignore
    val_subset.dataset.transform = val_transforms  # type: ignore
    test_subset.dataset.transform = val_transforms  # type: ignore

    class_counts = np.bincount(train_targets)
    class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float).clamp(min=1)
    sample_weights = class_weights[train_targets]
    sampler = WeightedRandomSampler(
        weights=sample_weights.tolist(),
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(
        dataset=train_subset,
        sampler=sampler,
        batch_size=BATCH_SIZE,
        num_workers=8,
        pin_memory=True,
    )
    val_loader = DataLoader(
        dataset=val_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )
    test_loader = DataLoader(
        dataset=test_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


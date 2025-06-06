import numpy as np
from typing import List, Tuple, Optional, Union
from .data_utils import Dataset, DataLoader

# CrossValidator class to perform k-fold cross-validation and other validation strategies


class CrossValidator:
    def __init__(self, validation_strategy: str = "k-fold", **kwargs) -> None:
        assert validation_strategy in ["k-fold", "leave-one-out", "stratified"],\
            "Validation strategy must be one of: 'k-fold', 'leave-one-out', 'stratified'"
        self.validation_strategy = validation_strategy
        self.kwargs = kwargs  # Store parameters like k, etc.
        self.random_state = np.random.RandomState(42)  # Add separate random state


    def get_folds(self, dataset: Dataset) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Returns list of (train_indices, val_indices) for the chosen strategy"""
        if self.validation_strategy == "k-fold":
            k = self.kwargs.get('k', 5)
            indices = np.arange(len(dataset))
            self.random_state.shuffle(indices)
            fold_size = len(dataset) // k
            folds = []
            for i in range(k):
                val_idx = indices[i*fold_size:(i+1)*fold_size]
                train_idx = np.concatenate([indices[:i*fold_size], 
                                          indices[(i+1)*fold_size:]])
                folds.append((train_idx, val_idx))
            return folds
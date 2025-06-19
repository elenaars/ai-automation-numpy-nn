"""K-fold cross-validation implementation.

This module provides a CrossValidator class that implements k-fold cross-validation
for splitting datasets into training and validation sets for model evaluation.
"""

import numpy as np
from typing import List, Tuple, Optional, Union
from .data_utils import Dataset, DataLoader


class CrossValidator:
    """K-fold cross-validation implementation.
    
    Args:
        k: Number of folds (default: 5)
        random_state: Random seed for reproducibility (default: 42)
    """
    def __init__(self, k: int = 5, random_state: int = 42) -> None:
        self.k = k
        self.random_state = np.random.RandomState(random_state)


    def get_folds(self, dataset: Dataset) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate k-fold cross-validation splits.
        
        Args:
            dataset: Dataset to split into folds
            
        Returns:
            List of (train_indices, val_indices) tuples for each fold
        """    
        
        indices = np.arange(len(dataset))
        self.random_state.shuffle(indices)
        fold_size = len(dataset) // self.k
        folds = []
            
        for i in range(self.k):
            val_idx = indices[i*fold_size:(i+1)*fold_size]
            train_idx = np.concatenate([indices[:i*fold_size], 
                                          indices[(i+1)*fold_size:]])
            folds.append((train_idx, val_idx))
        
        return folds
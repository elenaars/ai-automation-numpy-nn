"""
Utility functions for data processing.
including one-hot encoding of labels.
"""

import numpy as np

def one_hot_encode(labels: np.ndarray, num_classes: int = None) -> np.ndarray:
    """
    Convert any array of integer/string labels to one-hot.

    If labels are not 0…C-1, relabel them internally to that range.
    Input:
        labels: 1D array of labels (can be integers or strings)
        num_classes: Optional; if provided, specifies the number of classes.
                     If None, it will be inferred from the labels.
    Output:
        one_hot: 2D array of shape (n_samples, n_classes) with one-hot encoded labels.
    Raises:
        ValueError: If num_classes is less than the number of unique labels.
    """
    labels = np.asarray(labels)
    _, y_int = np.unique(labels, return_inverse=True)
    labels = y_int
    n = labels.shape[0]
    inferred_classes = labels.max() + 1
    if num_classes is None:
        num_classes = inferred_classes
    if num_classes < inferred_classes:
        raise ValueError(f"num_classes={num_classes} is less than the number of unique labels ({inferred_classes})")
    one_hot = np.zeros((n, num_classes), dtype=np.float32)
    one_hot[np.arange(n), labels] = 1.0
    return one_hot

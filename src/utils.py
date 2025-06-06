# miscellaneous utility functions

import numpy as np

def one_hot_encode(labels: np.ndarray, num_classes: int = None) -> np.ndarray:
    '''
    Convert any array of integer/string labels to one-hot.

    If labels are not 0…C-1, relabel them internally to that range.
    Input:
        labels: 1D array of labels (can be integers or strings)
        num_classes: Optional; if provided, specifies the number of classes.
                     If None, it will be inferred from the labels.
    Output:
        one_hot: 2D array of shape (n_samples, n_classes) with one-hot encoded labels.
    '''
    labels = np.asarray(labels)
    if labels.dtype.kind in ("U", "S", "O"):  # string or object
        # Use LabelEncoder‐style reindexing
        __, y_int = np.unique(labels, return_inverse=True)
        labels = y_int
    else:
        # If integer but not starting at zero or has gaps, remap:
        __, y_int = np.unique(labels, return_inverse=True)
        labels = y_int  # now in 0…(n_unique-1)

    n = labels.shape[0]
    if num_classes is None:
        num_classes = labels.max() + 1
    one_hot = np.zeros((n, num_classes), dtype=np.float32)
    one_hot[np.arange(n), labels] = 1.0
    return one_hot

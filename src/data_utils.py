"""
Utility functions for loading and processing datasets.
This module provides functions to load datasets from OpenML, generate synthetic datasets,
and normalize features. 

It also includes Dataset and DataLoader classes for handling data
and batching.
"""


from typing import Tuple, Optional, Union
import os

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler

# Dataset and DataLoader classes to wrap the data and operate on batches
class Dataset:
    """
    Dataset class to hold the data and labels.
    It provides methods to access the data and labels by index.
    """
    def __init__(self, x: np.ndarray, y: np.ndarray)-> None:
        self.x = x
        self.y = y

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: Union[int, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        assert isinstance(index, (int, np.ndarray)), "Index must be an integer or a numpy array."
        return self.x[index], self.y[index]
    
class DataLoader:
    """   
    DataLoader class to load the data in batches.
    It provides methods to iterate over the data in batches.
    """
    def __init__(self, dataset: Dataset, indices: Optional[np.ndarray] = None, batch_size: int = 32, shuffle: bool = False) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = indices if indices is not None else np.arange(len(dataset))
        self.current_index = 0

    def __iter__(self) -> 'DataLoader':
        self.current_index = 0
        if self.shuffle:
            np.random.shuffle(self.indices)
        return self

    def __next__(self)-> Tuple[np.ndarray, np.ndarray]:
        if self.current_index >= len(self.indices):
            raise StopIteration
        start_index = self.current_index
        end_index = min(start_index + self.batch_size, len(self.indices))
        batch_indices = self.indices[start_index:end_index]
        x_batch, y_batch = self.dataset[batch_indices]
        self.current_index += self.batch_size
        return x_batch, y_batch
    
    def __len__(self) -> int:
        return int(np.ceil(len(self.indices) / self.batch_size))
    
    @staticmethod
    def holdout_split(dataset: Dataset, test_size: float = 0.2, random_state: int = 42) -> Tuple['Dataset', 'Dataset']:
        """
        Splits the dataset into training and testing sets.
        Args:
            dataset (Dataset): The dataset to split.
            test_size (float): The proportion of the dataset to include in the test split.
        Returns:
            Tuple[Dataset, Dataset]: Training and testing datasets.
        """
        assert 0 < test_size < 1, "test_size must be between 0 and 1."
        indices = np.arange(len(dataset))
        np.random.RandomState(random_state).shuffle(indices)
        split_index = int(len(dataset) * (1 - test_size))
        train_indices = indices[:split_index]
        test_indices = indices[split_index:]
        return Dataset(dataset.x[train_indices], dataset.y[train_indices]), Dataset(dataset.x[test_indices], dataset.y[test_indices]) 
   

def generate_spiral_data(
    n_samples: int = 1500,
    n_classes: int = 3,
    class_sep: float = 0.2,
    random_state: int = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a 2D “spiral” synthetic classification dataset.

    Args:
        n_samples       Total number of points to generate (if not divisible by n_classes, the remainder is dropped).
        n_classes       Number of spiral arms (= number of classes).
        noise           Standard deviation of Gaussian noise added to the angle (controls “fuzziness” of each spiral).
        random_state    If provided, seed for NumPy’s RNG for reproducibility.

    Returns:
        X: np.ndarray of shape (n_used, 2)   # n_used = (n_samples // n_classes) * n_classes
           Each row is a 2D point on one of the spirals.
        y: np.ndarray of shape (n_used,)     # integer labels in {0, ... , n_classes-1}
    """
    if random_state is not None:
        np.random.seed(random_state)

    X = np.zeros((n_samples, 2))  # Ensure 2D data
    y = np.zeros(n_samples, dtype=int)
    
    # Determine how many points per class. Drop any remainder, 
    # so that we use exactly n_used = n_points_per_class * n_classes points.
    samples_per_class = n_samples // n_classes

    for class_idx in range(n_classes):
        ix = range(samples_per_class * class_idx, samples_per_class * (class_idx + 1))
        r = np.linspace(0.0, 1, samples_per_class) 
        t = np.linspace(class_idx * 4, (class_idx + 1) * 4, samples_per_class) + \
            np.random.randn(samples_per_class) * 0.2 * class_sep
        X[ix] = np.c_[r * np.sin(t * 2.5), r * np.cos(t * 2.5)]
        y[ix] = class_idx

    return X, y


def normalize_features(X: np.ndarray, dataset_name: str) -> np.ndarray:
    """
    Normalize features based on dataset characteristics. 
    If the dataset is MNIST or Fashion-MNIST, it normalizes to [0, 1].
    For Digits and Iris datasets, it uses StandardScaler.
    For synthetic or unknown datasets, it applies min-max normalization.
    
    Args:        
        X: Feature matrix (numpy array)
        dataset_name: Name of the dataset ('mnist', 'fashion_mnist', 'digits', 'iris')      
    Returns:
        Normalized feature matrix
    """
    
    if dataset_name in ['mnist', 'fashion_mnist']:
        # These datasets are already normalized to [0,1] by division by 255
        return X / 255.0 if X.max() > 1.0 else X
    
    elif dataset_name in ['digits', 'iris']:
        # Use StandardScaler for these datasets
        scaler = StandardScaler()
        return scaler.fit_transform(X)
    
    else:
        # For synthetic data or unknown datasets, use min-max normalization
        return (X - X.min()) / (X.max() - X.min())
    
def load_openml_dataset(dataset_name: str, data_dir: str = './data') -> Tuple[np.ndarray, np.ndarray]:
    """Generic function to load and cache OpenML datasets with proper normalization.
        Args:
            dataset_name: Name of the dataset ('mnist', 'fashion-mnist', 'iris')
            data_dir: Directory to store cached datasets
    
        Returns:
            Tuple of (X, y) arrays
    """
        
    # Mapping of friendly names to OpenML identifiers
    dataset_mapping = {
        'mnist': 'mnist_784',
        'fashion_mnist': 'Fashion-MNIST',
        'digits': 'digits',
        'iris': 'iris'
    }
    
    if dataset_name not in dataset_mapping:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from {list(dataset_mapping.keys())}")
    
    # Check if cached files exist
    cache_file = os.path.join(data_dir, f'{dataset_name}.npz')
    
    if os.path.exists(cache_file):
        try:
            print(f"Loading {dataset_name} from cache...")
            with np.load(cache_file) as data:
                return data['X'], data['y']
        except Exception as e:
            print(f"Cache loading failed: {e}. Removing corrupted cache and re-downloading...")
            os.remove(cache_file)
    
    print(f"Downloading {dataset_name} dataset...")
    dataset = fetch_openml(dataset_mapping[dataset_name], version=1)
    
    # Handle different return formats from fetch_openml
    X = dataset.data.values if hasattr(dataset.data, 'values') else dataset.data
    X = X.astype(np.float32)
    y = dataset.target.astype(np.int64)
    
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    X = normalize_features(X, dataset_name)
    
    # Save to cache
    print(f"Saving {dataset_name} to cache...")
    np.savez_compressed(cache_file, X=X, y=y)
    
    return X, y
            
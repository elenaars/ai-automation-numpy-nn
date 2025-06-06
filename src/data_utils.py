from typing import List, Tuple, Optional, Union, Dict, Any

import numpy as np
from sklearn.datasets import make_moons, fetch_openml


# Dataset and DataLoader classes to wrap the data and operate on batches
class Dataset:
    '''
    Dataset class to hold the data and labels.
    It provides methods to access the data and labels by index.
    '''
    def __init__(self, x: np.ndarray, y: np.ndarray)-> None:
        self.x = x
        self.y = y

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: Union[int, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        assert isinstance(index, (int, np.ndarray)), "Index must be an integer or a numpy array."
        return self.x[index], self.y[index]
    
class DataLoader:
    '''
    DataLoader class to load the data in batches.
    It provides methods to iterate over the data in batches.
    '''
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
    def holdout_split(dataset: Dataset, test_size: float = 0.2) -> Tuple['DataLoader', 'DataLoader']:
        """
        Splits the dataset into training and testing sets.
        Args:
            dataset (Dataset): The dataset to split.
            test_size (float): The proportion of the dataset to include in the test split.
        Returns:
            DataLoader: Loader for the training portion of the dataset.
            DataLoader: Loader for the testing portion of the dataset.
        """
        assert 0 < test_size < 1, "test_size must be between 0 and 1."
        indices = np.arange(len(dataset))
        np.random.shuffle(indices)
        split_index = int(len(dataset) * (1 - test_size))
        train_indices = indices[:split_index]
        test_indices = indices[split_index:]
        return DataLoader(dataset, train_indices), DataLoader(dataset, test_indices)
   
   
# functions to generate synthetic datasets
def generate_spiral_data(n_points_per_class: int, n_classes: int) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Generate spiral data for classification.
    n_points_per_class: number of points per class
    n_classes: number of classes
    return: tuple (X, y_one_hot)
    X: 2D array of shape (n_samples, 2) with the data points
    y_one_hot: 2D array of shape (n_samples, n_classes) with one-hot encoded labels
    '''
    x = []
    y = []
    for j in range(n_classes):
        ix = range(n_points_per_class * j, n_points_per_class * (j + 1))
        r = np.linspace(0.0, 1, n_points_per_class)
        t = np.linspace(j * 4, (j + 1) * 4, n_points_per_class) + np.random.randn(n_points_per_class) * 0.2
        x1 = r * np.sin(t)
        x2 = r * np.cos(t)
        x.append(np.c_[x1, x2])
        y.append(np.full(n_points_per_class, j))
    x = np.vstack(x)
    y = np.hstack(y)
    y_one_hot = np.eye(n_classes)[y]
    return x, y_one_hot

def generate_moons_data(n_samples: int, noise: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Generate two interleaving half circles (moons) for classification.
    n_samples: total number of samples
    noise: standard deviation of Gaussian noise added to the data
    return: tuple (X, y)
    X: 2D array of shape (n_samples, 2) with the data points
    y: 1D array of shape (n_samples,) with labels (0 or 1)
    '''

    X, y = make_moons(n_samples=n_samples, noise=noise)
    return X, y


#functions to download and return classic dataset
def download_mnist_data() -> Tuple[np.ndarray, np.ndarray]:
    '''
    Download the MNIST dataset and return the training and test data.
    return: tuple (X, y_one_hot)
    X: 2D array of shape (n_samples, 2) with the data points
    y_one_hot: 2D array of shape (n_samples, n_classes) with one-hot encoded labels
    '''
    mnist = fetch_openml('mnist_784', version=1)
    X = mnist.data.values.astype(np.float32) / 255.0  # Normalize to [0, 1]
    y = mnist.target.astype(np.int64)
    n_classes = 10
    y_one_hot = np.eye(n_classes)[y]
    return X, y_one_hot

#download and return fashion mnist dataset
def download_fashion_mnist_data() -> Tuple[np.ndarray, np.ndarray]:
    '''
    Download the Fashion MNIST dataset and return the training and test data.
    return: tuple (X, y_one_hot)
    X: 2D array of shape (n_samples, 2) with the data points
    y_one_hot: 2D array of shape (n_samples, n_classes) with one-hot encoded labels
    '''
    fashion_mnist = fetch_openml('Fashion-MNIST', version=1)
    X = fashion_mnist.data.values.astype(np.float32) / 255.0  # Normalize to [0, 1]
    y = fashion_mnist.target.astype(np.int64)
    n_classes = 10
    y_one_hot = np.eye(n_classes)[y]
    return X, y_one_hot

#download and return digits dataset
def download_digits_data() -> Tuple[np.ndarray, np.ndarray]:
    '''
    Download the Digits dataset and return the training and test data.
    return: tuple (X, y_one_hot)
    X: 2D array of shape (n_samples, 2) with the data points
    y_one_hot: 2D array of shape (n_samples, n_classes) with one-hot encoded labels
    '''
    digits = fetch_openml('mnist_784', version=1)
    X = digits.data.values.astype(np.float32) / 255.0  # Normalize to [0, 1]
    y = digits.target.astype(np.int64)
    n_classes = 10
    y_one_hot = np.eye(n_classes)[y]
    return X, y_one_hot

#download and return iris dataset
def download_iris_data() -> Tuple[np.ndarray, np.ndarray]:
    '''
    Download the Iris dataset and return the data and labels.
    return: tuple (X, y_one_hot)
    X: 2D array of shape (n_samples, n_features) with the data points
    y_one_hot: 2D array of shape (n_samples, n_classes) with one-hot encoded labels
    '''
    iris = fetch_openml('iris', version=1)
    X = iris.data.values.astype(np.float32)
    y = iris.target.astype(np.int64)
    n_classes = len(np.unique(y))
    y_one_hot = np.eye(n_classes)[y]
    return X, y_one_hot
 
from typing import Tuple, Optional, Union

import numpy as np
from sklearn.datasets import fetch_openml



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
    def holdout_split(dataset: Dataset, test_size: float = 0.2, batch_size: int = 32) -> Tuple['Dataset', 'Dataset']:
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
        #return DataLoader(dataset, train_indices, batch_size=batch_size, shuffle = True), DataLoader(dataset, test_indices, batch_size=batch_size, shuffle = False)
        return Dataset(dataset.x[train_indices], dataset.y[train_indices]), Dataset(dataset.x[test_indices], dataset.y[test_indices]) 
   
# function to generate synthetic dataset - spiral.

def generate_spiral_data(
    n_samples: int = 1500,
    n_classes: int = 3,
    noise: float = 0.2,
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

    # Determine how many points per class. Drop any remainder.
    n_points_per_class = n_samples // n_classes
    #n_used = n_points_per_class * n_classes

    x_list = []
    y_list = []

    for class_idx in range(n_classes):
        # r goes from 0.0 out to 1.0 (radial coordinate)
        r = np.linspace(0.0, 1.0, n_points_per_class)
        # theta sweeps out 4pi per class (so the spirals loop around twice),
        # offset by class_idx*4 so each class has its own arm.
        t = np.linspace(class_idx * 4.0, (class_idx + 1) * 4.0, n_points_per_class) 
        # Add Gaussian “noise” to the angle t
        t = t + np.random.randn(n_points_per_class) * noise

        # Convert (r, t) to Cartesian coordinates
        x1 = r * np.sin(t * np.pi / 2)  # optionally -- multiply t by pi/2 for a tighter spiral
        x2 = r * np.cos(t * np.pi / 2)

        # Stack into (n_points_per_class, 2)
        x_class = np.column_stack([x1, x2])
        y_class = np.full(n_points_per_class, class_idx, dtype=np.int64)

        x_list.append(x_class)
        y_list.append(y_class)

    # Concatenate all classes
    X = np.vstack(x_list)   # shape: (n_used, 2)
    y = np.hstack(y_list)   # shape: (n_used,)

    return X, y

#download and return mnist dataset
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
    return X, y

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
    return X, y

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
    return X, y

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
    return X, y
 
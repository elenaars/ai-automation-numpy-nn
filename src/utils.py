import numpy as np
from typing import List, Tuple, Optional, Union
from sklearn.datasets import make_moons, fetch_openml



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

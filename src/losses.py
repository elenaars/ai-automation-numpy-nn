""" 
This module defines several loss functions used in machine learning models.
It includes abstract base class Loss and concrete implementations for Mean Squared Error and Cross Entropy with Softmax.
"""

from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np


def softmax(x: np.ndarray) -> np.ndarray:
    """
    Compute softmax values for each sets of scores in x.
    The softmax function is defined as:
    softmax(x_i) = exp(x_i) / sum(exp(x_j))
    where x_i is the i-th element of the input vector x and the sum is over all elements in x.

    Args:
        x: 2D array of shape (n_samples, n_classes)
    Returns: 
        2D array of shape (n_samples, n_classes) with softmax probabilities
    
    """
    
    # Subtract max for numerical stability
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


class Loss(ABC):
    """
    Abstract base class for all loss functions.
    Each loss function should implement the forward and backward methods.
    """
    
    @abstractmethod
    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Forward pass through the loss function.
        Args:
            y_true (np.ndarray): True labels. The shape should be (batch_size,).
            y_pred (np.ndarray): Predicted labels. The shape should be (batch_size,).
        Returns:
            tuple: A tuple containing the loss value and the predicted probabilities.
            The first element is a scalar (loss value), and the second element is an array of shape (batch_size,).
        """
        pass

    @abstractmethod
    def backward(self, y_true: np.ndarray, y_pred: np.ndarray, probs: np.ndarray) -> np.ndarray:
        """
        Backward pass through the loss function.
        Args:
            y_true (np.ndarray): True labels. The shape should be (batch_size,).
            y_pred (np.ndarray): Predicted labels. The shape should be (batch_size,).
            probs (np.ndarray): Predicted probabilities. The shape should be (batch_size,).
        Returns:
            np.ndarray: Gradient of the loss with respect to the predictions. The shape should be (batch_size,).
        """
        pass
    
    
# Define concrete implementation of Loss: MeanSquaredError and CrossEntropy

class MeanSquaredError(Loss):
    """
    Mean Squared Error (MSE) loss function.
    It measures the average squared difference between the predicted and true values.
    The MSE is defined as:
    MSE = (1/n) * sum((y_true - y_pred)^2)
    where n is the number of samples, y_true is the true labels, and y_pred is the predicted labels.
    """
    
    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, np.ndarray]:
        """ 
        Forward pass through the mean squared error loss function.      
        Args:
            y_true (np.ndarray): True labels. The shape should be (batch_size,).
            y_pred (np.ndarray): Predicted labels. The shape should be (batch_size,).
        Returns:
            tuple: A tuple containing the loss value and the predicted labels.
            The first element is a scalar (loss value), and the second element is an array of shape (batch_size,).
        """
        if y_true.shape != y_pred.shape:
            raise ValueError(f"Shape mismatch in the meansquare error coaculation: true labels {y_true.shape} and predicted labels {y_pred.shape} must have the same shape.")
        return np.mean(np.square(y_true - y_pred)), y_pred

    def backward(self, y_true: np.ndarray, y_pred: np.ndarray, probs: np.ndarray = None) -> np.ndarray:
        """
        Backward pass through the mean squared error loss function.
        Args:
            y_true (np.ndarray): True labels. The shape should be (batch_size,).
            y_pred (np.ndarray): Predicted labels. The shape should be (batch_size,).
            probs (np.ndarray): Not used in this loss function, but included for consistency.       
        Returns:
            np.ndarray: Gradient of the loss with respect to the predictions. The shape should be (batch_size,).
        """
        
        if y_true.shape != y_pred.shape:
            raise ValueError(f"Shape mismatch in the meansquare error coaculation: true labels {y_true.shape} and predicted labels {y_pred.shape} must have the same shape.")
        return 2 * (y_pred - y_true) / y_true.size
    
    
class CrossEntropySoftMax(Loss):
    """
    Cross-entropy loss function with softmax activation.
    It is used for multi-class classification problems.
    The cross-entropy loss is defined as:
    CE = -sum(y_true * log(probs))
    where y_true is the one-hot encoded true labels and probs is the predicted probabilities.
    """
    
    def forward(self, y_true: np.ndarray, y_pred_logits: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Forward pass through the cross-entropy loss function.
        Args:
            y_true (np.ndarray): True labels (one-hot encoded). The shape should be (batch_size, n_classes).
            y_pred_logits (np.ndarray): Predicted logits. The shape should be (batch_size, n_classes).
        Returns:
            tuple: A tuple containing the loss value and the predicted probabilities.
            The first element is a scalar (loss value), and the second element is an array of shape (batch_size, n_classes).
        """
        if y_true.shape != y_pred_logits.shape:
            raise ValueError(f"Shape mismatch in the cross-entropy loss calculation: true labels {y_true.shape} and predicted logits {y_pred_logits.shape} must have the same shape.")
        #apply softmax to the predictions
        probs = softmax(y_pred_logits)
        loss = -np.sum(y_true * np.log(probs + 1e-15)) / y_true.shape[0]
        return loss, probs

    def backward(self, y_true: np.ndarray, y_pred: np.ndarray, probs: np.ndarray) -> np.ndarray:
        """
        Backward pass through the cross-entropy loss function.
        Args:
            y_true (np.ndarray): True labels (one-hot encoded). The shape should be (batch_size, n_classes).
            y_pred (np.ndarray): Predicted logits. The shape should be (batch_size, n_classes). Here for consistency, but not used in this loss.
            probs (np.ndarray): Predicted probabilities. The shape should be (batch_size, n_classes).
        Returns:
            np.ndarray: Gradient of the loss with respect to the predictions. The shape should be (batch_size, n_classes).
        """
        
        if y_true.shape != probs.shape:
            raise ValueError(f"Shape mismatch in the cross-entropy loss backward pass: true labels {y_true.shape} and predicted probabilities {probs.shape} must have the same shape.")
        return (probs - y_true)/y_true.shape[0]
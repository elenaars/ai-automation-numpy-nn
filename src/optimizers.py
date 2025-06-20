"""
This module defines an abstract base class for optimizers and a concrete implementation of Stochastic Gradient Descent (SGD).
The Optimizer class provides an interface for updating model parameters based on gradients,
while the SGD class implements the SGD optimization algorithm.
"""

from abc import ABC, abstractmethod
import numpy as np


class Optimizer(ABC):
    @abstractmethod
    def step(self, params: np.ndarray, grads: np.ndarray) -> None:
        """
        Update the parameters based on the gradients.
        Args:
            params (np.ndarray): Parameters to be updated. The shape should be (num_params,).
            grads (np.ndarray): Gradients of the loss with respect to the parameters. The shape should be (num_params,).
        Returns:
            None
        """
        pass
    
    def update_learning_rate(self, new_lr: float) -> None:
        """
        Update the learning rate of the optimizer.
        Args:
            new_lr (float): New learning rate.
        Returns:
            None
        """
        if new_lr <= 0:
            raise ValueError(f"Optimizer: Learning rate must be a positive number, got {new_lr}")
        self.learning_rate = new_lr
        
# Define concrete implementation of Optimizer: SGD

class SGD(Optimizer):
    """
    Stochastic Gradient Descent (SGD) optimizer.
    It updates the parameters using the gradients and a learning rate.
    The update rule is defined as:
    params = params - learning_rate * grads
    where params are the parameters to be updated, learning_rate is the learning rate, and grads are the gradients.
    """
    
    def __init__(self, learning_rate: float = 0.01) -> None:
        if learning_rate <= 0:
            raise ValueError(f"SGD: Learning rate must be a positive number, got {learning_rate}")
        self.learning_rate = learning_rate

    def step(self, params: np.ndarray, grads: np.ndarray) -> None:
        if params.shape != grads.shape:
            raise ValueError(f"SGD: Shape mismatch in SGD step: parameters {params.shape} and gradients {grads.shape} must have the same shape.")
        params -= self.learning_rate * grads
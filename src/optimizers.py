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
    Stochastic Gradient Descent (SGD) optimizer with optional momentum and weight decay.

    This optimizer maintains a separate velocity vector for each parameter array.
    The update rule is:
        v = momentum * v - learning_rate * (grads + weight_decay * params)
        params += v

    Args:
        learning_rate (float): Learning rate for parameter updates.
        momentum (float): Momentum factor (default: 0.0, no momentum).
        weight_decay (float): L2 regularization factor (default: 0.0, no weight decay).
    """
    
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.0, weight_decay: float = 0.0) -> None:
        if learning_rate <= 0:
            raise ValueError(f"SGD: Learning rate must be a positive number, got {learning_rate}")
        if momentum < 0 or momentum >= 1:
            raise ValueError(f"SGD: Momentum must be in [0, 1), got {momentum}")
        if weight_decay < 0:
            raise ValueError(f"SGD: Weight decay must be non-negative, got {weight_decay}")
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocity = {}  # Dict to store velocity for each param. Will be initialized on first step

    def step(self, params: np.ndarray, grads: np.ndarray) -> None:
        """
        Update the parameters using SGD with momentum and weight decay.

        Maintains a separate velocity for each parameter array.

        Args:
            params (np.ndarray): Parameters to be updated.
            grads (np.ndarray): Gradients of the loss with respect to the parameters.

        Returns:
            None
        """
        
        if params.shape != grads.shape:
            raise ValueError(f"SGD: Shape mismatch in SGD step: parameters {params.shape} and gradients {grads.shape} must have the same shape.")
        param_id = id(params)
        if param_id not in self.velocity:
            self.velocity[param_id] = np.zeros_like(params)
        # Apply weight decay
        grads += self.weight_decay * params
        # Update velocity with momentum
        self.velocity[param_id] = self.momentum * self.velocity[param_id] - self.learning_rate * grads
        params += self.velocity[param_id]
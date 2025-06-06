from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Union, Dict, Any

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
        self.learning_rate = new_lr
        
# Define concrete implementation of Optimizer: SGD

class SGD(Optimizer):
    '''
    Stochastic Gradient Descent (SGD) optimizer.
    It updates the parameters using the gradients and a learning rate.
    The update rule is defined as:
    params = params - learning_rate * grads
    where params are the parameters to be updated, learning_rate is the learning rate, and grads are the gradients.
    '''
    
    def __init__(self, learning_rate: float = 0.01) -> None:
        assert learning_rate > 0, "Learning rate must be a positive number."
        self.learning_rate = learning_rate

    def step(self, params: np.ndarray, grads: np.ndarray) -> None:
        assert params.shape == grads.shape, f"Parameters shape {params.shape} does not match gradients shape {grads.shape}"
        params -= self.learning_rate * grads
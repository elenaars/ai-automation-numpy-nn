"""
This module defines the Layer class and its concrete implementations for a neural network.
It includes Linear, ReLU, and Sequential layers, each implementing the forward and backward methods.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Union, Dict, Any
import numpy as np
import logging


class Layer(ABC):
    """
    Abstract base class for all layers in the neural network.
    Each layer should implement the forward and backward methods, 
    the instances store their input and output dimensions.
    """
    
    def __init__(self, input_dim: Optional[int] = None, output_dim: Optional[int] = None, logger: Optional[logging.Logger] = None) -> None:
        self._input_dim = input_dim
        self._output_dim = output_dim
        self.input = None
        self.logger = logger

    @property
    def input_dim(self) -> Optional[int]:
        return self._input_dim

    @property
    def output_dim(self) -> Optional[int]:
        return self._output_dim
    
    
    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through the layer.
        Args:
            x (np.ndarray): Input array of shape (batch_size, input_dim).
        Returns:
            np.ndarray: Output array of shape (batch_size, output_dim).
        """
        pass

    @abstractmethod
    def backward(self, grad: np.ndarray) -> np.ndarray:
        """
        Backward pass through the layer.
        Args:
            grad (np.ndarray): Gradient of the loss with respect to the output. 
            The shape should be (batch_size, output_dim).
        Returns:
            np.ndarray: Gradient of the loss with respect to the input.
            The shape should be (batch_size, input_dim).
        """
        pass
    
    
# Define concrete implementation of Layer: Linear, ReLU, and Sequential

class ReLU(Layer):
    """
    ReLU layer.
    Applies the ReLU activation function element-wise to the input.
    The ReLU function is defined as f(x) = max(0, x).
    """
    def __init__(self, logger: logging.Logger) -> None:
        super().__init__(logger = logger)
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        return np.maximum(0, x)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        if grad.shape != self.input.shape:
            raise ValueError(f"ReLU layer: Gradient shape {grad.shape} does not match input shape {self.input.shape}")
        # Gradient of ReLU is 1 for positive inputs, 0 for negative inputs
        return grad * (self.input > 0)
    
    
class Linear(Layer):
    """
    Linear layer.
    Applies a linear transformation to the input data.
    The transformation is defined as y = xW + b, where W is the weight matrix and b is the bias vector.
    For weights, we use He initialization to ensure that the weights are initialized
    in a way that maintains the variance of the activations across layers.
    The bias is initialized to zeros.
    """
    def __init__(self, input_dim: int, output_dim: int, logger: logging.Logger) -> None:
        if input_dim < 1 or output_dim < 1:
            raise ValueError("input_dim and output_dim must be positive integers.")
        super().__init__(input_dim, output_dim, logger = logger)
        # Initialize weights and bias
        self.weights = np.random.randn(input_dim, output_dim) * np.sqrt(2.0 / input_dim)
        self.bias = np.zeros((1, output_dim))
        self.grad_weights = None
        self.grad_bias = None   

    def forward(self, x: np.ndarray) -> np.ndarray:
        if x.shape[1] != self.weights.shape[0]:
            raise ValueError(f"Input shape {x.shape} does not match expected shape (batch_size, {self.weights.shape[0]})")
        if self.weights.shape[1] != self.bias.shape[1]:
            raise ValueError(f"Weights shape {self.weights.shape} does not match bias shape {self.bias.shape}")
        self.input = x
        return x @ self.weights + self.bias

    def backward(self, grad: np.ndarray) -> np.ndarray:
        if grad.shape[1] != self.bias.shape[1]:
            raise ValueError(f"Gradient shape {grad.shape} does not match bias shape {self.bias.shape}")
        if grad.shape[0] != self.input.shape[0]:
            raise ValueError(f"Gradient shape {grad.shape} does not match input shape {self.input.shape}")

        # Gradient of the loss with respect to the input
        grad_input = grad @ self.weights.T
        # Gradient of the loss with respect to the weights and bias
        self.grad_weights = self.input.T @ grad
        self.grad_bias = np.sum(grad, axis=0, keepdims=True)
              
        # Monitor gradient statistics
        grad_stats = {
            'input_grad_max': np.abs(grad_input).max(),
            'weight_grad_max': np.abs(self.grad_weights).max(),
            'bias_grad_max': np.abs(self.grad_bias).max()
        }
    
        if any(np.isnan(v) or np.isinf(v) for v in grad_stats.values()):
            raise ValueError(f"Invalid gradients detected: {grad_stats}")
        
        return grad_input 
    
class Sequential(Layer):
    """
    Sequential model.
    A container for stacking layers in a linear fashion.
    The input to the first layer is the input to the model, and the output of the last layer is the output of the model.
    """
    def __init__(self, layers: List[Layer], logger: logging.Logger) -> None:
        if len(layers) < 1:
            raise ValueError("Sequential model must have at least one layer.")
        self.layers = layers
        super().__init__(layers[0].input_dim, layers[-1].output_dim, logger=logger)
        self.__check_consistency__()

    def __check_consistency__(self) -> None:
        if self.layers[0].input_dim is None:
            raise ValueError("First layer input dimension must be specified.")
        if self.layers[-1].output_dim is None:
            raise ValueError("Last layer output dimension must be specified.")
        if self.layers[0].input_dim != self.input_dim:
            raise ValueError(f"First layer input dimension {self.layers[0].input_dim} does not match expected input dimension {self.input_dim}")
        if self.layers[-1].output_dim != self.output_dim:
            raise ValueError(f"Last layer output dimension {self.layers[-1].output_dim} does not match expected output dimension {self.output_dim}")
        current_dim = self.input_dim
        mismatch_list = []
        for layer in self.layers:
            if layer.input_dim != None:
                if layer.input_dim != current_dim: 
                    mismatch_list.append(f"Layer {layer.__class__.__name__} input dimension {layer.input_dim} does not match expected input dimension {current_dim}")
                current_dim = layer.output_dim
        if len(mismatch_list) > 0:
            raise ValueError(f"Layer dimension mismatch: {'; '.join(mismatch_list)}")
        
                                
    def forward(self, x: np.ndarray) -> np.ndarray:
        if x.shape[1] != self.layers[0].input_dim:
            raise ValueError(f"Sequential: Input shape {x.shape} does not match expected shape (batch_size, {self.layers[0].input_dim})")

        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad: np.ndarray) -> np.ndarray:
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad
    
    def summary(self) -> None:
        """
        Log a summary of the model architecture.
        """
        self.logger.info("Model Summary: \n " + "-" * 50)
        total_params = 0
        for i, layer in enumerate(self.layers):
            if isinstance(layer, Linear):
                params = np.prod(layer.weights.shape) + np.prod(layer.bias.shape)
                total_params += params
                self.logger.info(f"Layer {i}: {layer.__class__.__name__}, "
                      f"Input: {layer.input_dim}, Output: {layer.output_dim}, "
                      f"Parameters: {params}")
            else:
                self.logger.info(f"Layer {i}: {layer.__class__.__name__}")
        #print("-" * 50)
        self.logger.info("-"*50 + f"Total parameters: {total_params}" + "\n" + "-" * 50)
        #print("-" * 50) 
        
    def save_architecture(self, file_path: str) -> None:
        """
        Save the model architecture to a file.
        Args:
            file_path (str): Path to the file where the architecture will be saved.
        """
        with open(file_path, 'w') as f:
            f.write("Model Architecture:\n")
            f.write("-" * 50 + "\n")
            for i, layer in enumerate(self.layers):
                if isinstance(layer, Linear):
                    f.write(f"Layer {i}: {layer.__class__.__name__}, "
                            f"Input: {layer.input_dim}, Output: {layer.output_dim}\n")
                else:
                    f.write(f"Layer {i}: {layer.__class__.__name__}\n")
            f.write("-" * 50 + "\n")
            f.write(f"Total parameters: {sum(np.prod(layer.weights.shape) + np.prod(layer.bias.shape) for layer in self.layers if isinstance(layer, Linear))}\n")
            f.write("-" * 50 + "\n")
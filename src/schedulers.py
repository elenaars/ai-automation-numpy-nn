"""Learning Rate Schedulers
This module defines various learning rate schedulers for training neural networks.
It includes schedulers for step decay, exponential decay, warmup, and cosine annealing.
"""

from abc import ABC, abstractmethod
import numpy as np

# Learning rate scheduler

class LRScheduler(ABC):
    """
    Abstract base class for learning rate schedulers.
    Each scheduler should implement the get_lr method to compute the current learning rate.
    """
    def __init__(self, initial_lr: float) -> None:
        if initial_lr <= 0:
            raise ValueError(f"LRScheduler: Initial learning rate must be a positive number, got {initial_lr}")
        self.initial_lr = initial_lr
        self.current_step = 0
        
    @abstractmethod
    def get_lr(self) -> float:
        """Calculate the learning rate for the current step"""
        pass
    
    def step(self) -> None:
        """Increment the step counter"""
        self.current_step += 1

class StepLRScheduler(LRScheduler):
    """Decays learning rate by gamma every step_size epochs"""
    def __init__(self, initial_lr: float, step_size: int, gamma: float = 0.1):
        if step_size <= 0:
            raise ValueError(f"StepLRScheduler: step_size must be a positive integer, got {step_size}")
        if gamma <= 0:
            raise ValueError(f"StepLRScheduler: gamma must be a positive number, got {gamma}")
        if initial_lr <= 0:
            raise ValueError(f"StepLRScheduler: Initial learning rate must be a positive number, got {initial_lr}")
        
        super().__init__(initial_lr)
        self.step_size = step_size
        self.gamma = gamma
        
    def get_lr(self) -> float:
        return self.initial_lr * (self.gamma ** (self.current_step // self.step_size))

class ExponentialLRScheduler(LRScheduler):
    """Exponentially decays learning rate by gamma every epoch"""
    def __init__(self, initial_lr: float, gamma: float = 0.95)-> None:
        if gamma <= 0:
            raise ValueError(f"ExponentialLRScheduler: gamma must be a positive number, got {gamma}")
        if initial_lr <= 0:
            raise ValueError(f"ExponentialLRScheduler: Initial learning rate must be a positive number, got {initial_lr}")
        super().__init__(initial_lr)
        self.gamma = gamma
        
    def get_lr(self) -> float:
        return self.initial_lr * (self.gamma ** self.current_step)
    
class WarmupLRScheduler(LRScheduler):
    """Warmup learning rate scheduler with cosine annealing after warmup"""
    def __init__(self, initial_lr: float, warmup_epochs: int = 50, total_epochs: int = 1000) -> None:
        if warmup_epochs < 0:
            raise ValueError(f"WarmupLRScheduler: warmup_epochs must be a non-negative integer, got {warmup_epochs}")
        if warmup_epochs >= total_epochs:
            raise ValueError(f"WarmupLRScheduler: warmup_epochs ({warmup_epochs}) must be less than total_epochs ({total_epochs})")
        super().__init__(initial_lr)
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
                
    def get_lr(self) -> float:
        if self.current_step < self.warmup_epochs:
            return self.initial_lr * (self.current_step / self.warmup_epochs)
        return self.initial_lr * 0.5 * (1 + np.cos(np.pi * (self.current_step - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)))
    
class CosineAnnealingLRScheduler(LRScheduler):
    """Cosine annealing scheduler with warmup and restarts. \
        Warmup is optional and is controlled by warmup_epochs and warmup_start_lr."""
    def __init__(self, initial_lr: float, T_max: int = 1000, eta_min: float = 0.0, warmup_epochs: int = 0, warmup_start_lr: float = 0.0001) -> None:
        if T_max <= 0:
            raise ValueError(f"CosineAnnealingLRScheduler: T_max must be a positive integer, got {T_max}")
        if eta_min < 0:
            raise ValueError(f"CosineAnnealingLRScheduler: eta_min must be a non-negative number, got {eta_min}")
        if warmup_epochs < 0:
            raise ValueError(f"CosineAnnealingLRScheduler: warmup_epochs must be a non-negative integer, got {warmup_epochs}")
        if warmup_start_lr < 0:
            raise ValueError(f"CosineAnnealingLRScheduler: warmup_start_lr must be a non-negative number, got {warmup_start_lr}")
        super().__init__(initial_lr)
        self.T_max = T_max
        self.eta_min = eta_min
        self.warmup_epochs = warmup_epochs
        self.warmup_start_lr = warmup_start_lr
        
    def get_lr(self) -> float:
        # Warmup phase
        if self.current_step < self.warmup_epochs:
            return self.warmup_start_lr + (self.initial_lr - self.warmup_start_lr) * \
                   (self.current_step / self.warmup_epochs)
        
        # Cosine annealing phase
        progress = (self.current_step - self.warmup_epochs) / (self.T_max - self.warmup_epochs)
        progress = min(1.0, progress)  # Clip progress to [0, 1]
        
        return self.eta_min + (self.initial_lr - self.eta_min) * \
               (1 + np.cos(np.pi * progress)) / 2
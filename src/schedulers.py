from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Union, Dict, Any

import numpy as np

# Learning rate scheduler

class LRScheduler(ABC):
    '''
    Abstract base class for learning rate schedulers.
    Each scheduler should implement the get_lr method to compute the current learning rate.
    '''
    def __init__(self, initial_lr: float) -> None:
        assert initial_lr > 0, "Initial learning rate must be positive"
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
        super().__init__(initial_lr)
        self.step_size = step_size
        self.gamma = gamma
        
    def get_lr(self) -> float:
        return self.initial_lr * (self.gamma ** (self.current_step // self.step_size))

class ExponentialLRScheduler(LRScheduler):
    """Exponentially decays learning rate by gamma every epoch"""
    def __init__(self, initial_lr: float, gamma: float = 0.95)-> None:
        super().__init__(initial_lr)
        self.gamma = gamma
        
    def get_lr(self) -> float:
        return self.initial_lr * (self.gamma ** self.current_step)
    
class WarmupLRScheduler(LRScheduler):
    def __init__(self, initial_lr: float, warmup_epochs: int = 50) -> None:
        super().__init__(initial_lr)
        self.warmup_epochs = warmup_epochs
        
    def get_lr(self) -> float:
        if self.current_step < self.warmup_epochs:
            return self.initial_lr * (self.current_step / self.warmup_epochs)
        return self.initial_lr * 0.5 * (1 + np.cos(np.pi * (self.current_step - self.warmup_epochs) / (1000 - self.warmup_epochs)))
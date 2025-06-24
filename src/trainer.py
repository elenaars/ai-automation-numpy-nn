"""
This module defines the Trainer class, which is responsible for training a machine learning model.
It includes methods for training with validation, computing accuracy, and performing k-fold cross-validation.
It also provides functionality for visualizing training metrics and decision boundaries.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import time
import copy
import logging
from typing import Dict, List, Union, Optional
from .layers import Linear, Sequential
from .optimizers import Optimizer
from .losses import Loss
from .data_utils import DataLoader, Dataset
from .visualizer import TrainingVisualizer, KFoldVisualizer
from .schedulers import LRScheduler, ExponentialLRScheduler
from .optimizers import SGD
from .cross_validator import CrossValidator

logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, model: Sequential, loss_fn: Loss, optimizer: Optimizer, exp_dir: str) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        
         # Setup experiment directories
        self.exp_dir = exp_dir
        self.plots_dir = os.path.join(self.exp_dir, 'plots')
        os.makedirs(self.exp_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)      

        
    def save_epoch_plots(self, train_loader: DataLoader, val_loader: DataLoader, epoch_dir: str) -> None:
        """
        Save plots for the current epoch.
        Args:
            train_loader (DataLoader): The DataLoader for the training set.
            val_loader (DataLoader): The DataLoader for the validation set.
            epoch_dir (str): Directory to save the plots for the current epoch.
        """
        self.visualizer.plot_decision_boundary(
            self.model, 
            train_loader.dataset.x, 
            train_loader.dataset.y,
            filepath=os.path.join(epoch_dir, "decision_boundary.png") 
            )
        
        self.visualizer.plot_loss_landscape(
            self.model,
            val_loader,
            self.loss_fn,
            filepath=os.path.join(epoch_dir,"loss_landscape.png")
            )
        
        self.visualizer.weights_gradients_heatmap(
            self.model,
            self.optimizer,
            filepath=os.path.join(epoch_dir,"weights_heatmap.png")
            )
        
        
    def train(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int = 1000,  show_plots_logs:bool = True, log_interval:int = 200, patience: int = 20, min_delta: float = 1e-4, lr_scheduler: Optional[LRScheduler] = None, **kwargs) -> Dict[str, List[float]]:
        """
        Train the model using the specified loss function and optimizer.

        The training loop performs:
            - Forward and backward passes for each batch in the training set.
            - Parameter updates using the optimizer.
            - Learning rate scheduling (if provided).
            - Validation and accuracy computation at the end of each epoch.
            - Early stopping based on validation loss improvement.
            - Optional logging and plotting at specified intervals.

        Args:
            train_loader (DataLoader): DataLoader for the training set.
            val_loader (DataLoader): DataLoader for the validation set.
            epochs (int): Number of epochs to train. Default is 1000.
            show_plots_logs (bool): Whether to save plots and log statistics during training. Default is True.
            log_interval (int): Interval (in epochs) for saving plots and logging. Default is 200.
            patience (int): Number of epochs to wait for improvement before early stopping. Default is 20.
            min_delta (float): Minimum relative improvement in validation loss to reset patience. Default is 1e-4.
            lr_scheduler (Optional[LRScheduler]): Learning rate scheduler. If None, uses ExponentialLRScheduler.
            **kwargs: Additional arguments (e.g., batch_size).

        Returns:
            Dict[str, List[float]]: Training history with keys:
                - 'loss': Training loss per epoch.
                - 'val_loss': Validation loss per epoch.
                - 'train_acc': Training accuracy per epoch.
                - 'val_acc': Validation accuracy per epoch.

        Raises:
            ValueError: If loss, validation loss, or accuracy are invalid (NaN, inf, or out of range).
        """
        
        def validate_metrics(loss: float, val_loss: float, acc: float) -> None:
            """Helper function to validate metrics"""
            if np.isnan(loss) or np.isinf(loss):
                raise ValueError(f"Training loss is {loss}")
            if np.isnan(val_loss) or np.isinf(val_loss):
                raise ValueError(f"Validation loss is {val_loss}")
            if acc < 0 or acc > 100:
                raise ValueError(f"Invalid accuracy value: {acc}")
        
        # If no scheduler provided, create default one
        if lr_scheduler is None:
            lr_scheduler = ExponentialLRScheduler(0.01, gamma=0.99)
            
        epoch_times = [] # List to store training times per epoch for logging
        
        # Initialize best validation loss, best model state and patiance counter for early stopping
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        
        # Training loop
        for epoch in range(epochs):
            
            start_time = time.time()
            epoch_loss = 0
            n_samples = 0
            
            current_lr = lr_scheduler.get_lr()           
            self.optimizer.update_learning_rate(current_lr)
            
            # Monitor gradients before updates
            max_grad = 0
            
            for x_batch, y_batch in train_loader:
                
                # Forward pass
                y_pred = self.model.forward(x_batch)
                
                # Compute loss
                loss, probs = self.loss_fn.forward(y_batch, y_pred)
                
                # Check for valid predictions
                if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
                    raise ValueError("Invalid predictions detected")
                
                epoch_loss += loss * x_batch.shape[0]
                n_samples += x_batch.shape[0]
                
                # Backward pass with gradient clipping
                grad = self.loss_fn.backward(y_batch, y_pred, probs)
                max_grad = max(max_grad, np.abs(grad).max())
                grad = np.clip(grad, -1.0, 1.0)
                self.model.backward(grad)
                
                # Update parameters
                for layer in self.model.layers:
                    if isinstance(layer, Linear):
                        if np.any(np.isnan(layer.grad_weights)) or np.any(np.isinf(layer.grad_weights)):
                            raise ValueError("Invalid gradients detected")
                        layer.grad_weights = np.clip(layer.grad_weights, -1.0, 1.0)
                        layer.grad_bias = np.clip(layer.grad_bias, -1.0, 1.0)
                        self.optimizer.step(layer.weights, layer.grad_weights)
                        self.optimizer.step(layer.bias, layer.grad_bias)
            
            #Step the learning rate scheduler
            lr_scheduler.step()
            
            epoch_loss /= n_samples
            
            # compute and validate the metrics for this epoch
            val_loss = self.validate(val_loader)
            train_acc = self.compute_accuracy(train_loader)
            val_acc = self.compute_accuracy(val_loader)
            
            try:
                validate_metrics(epoch_loss, val_loss, train_acc)
            except ValueError as e:
                logger.error(f"Error at epoch {epoch}: \n Max gradient magnitude: {max_grad} \n Current learning rate: {current_lr} ")
                raise e
            
            self.visualizer.update(epoch_loss, val_loss, train_acc, val_acc)
            this_epoch_time = time.time() - start_time
            epoch_times.append(this_epoch_time)

           
            # Early stopping check
            if val_loss < best_val_loss*(1 - min_delta):
                best_val_loss = val_loss
                best_model_state = copy.deepcopy([(layer.weights.copy(), layer.bias.copy()) 
                        for layer in self.model.layers 
                        if isinstance(layer, Linear)])
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch} epochs")
                # Restore best model
                linear_layers = [l for l in self.model.layers if isinstance(l, Linear)]
                for layer, (weights, bias) in zip(linear_layers, best_model_state):
                    layer.weights = weights.copy()
                    layer.bias = bias.copy()
                    
                # Truncate history at early stopping point
                for key in self.visualizer.history:
                    self.visualizer.history[key] = self.visualizer.history[key][:epoch + 1]
                
                break
            
            # Plot the decision boundary and loss landscape every log_interval epochs, but also at the last epoch
            if show_plots_logs and ((epoch % log_interval == 0 and epoch > 0) or epoch == epochs - 1):
                
                epoch_dir = os.path.join(self.visualizer.exp_dir, f'epoch_{epoch}')
                os.makedirs(epoch_dir, exist_ok=True)
                os.sync()
                
                self.save_epoch_plots(train_loader, val_loader, epoch_dir)
            
                logger.info(f"Saved plots for epoch {epoch} in {epoch_dir}")               
                    
                # Log essential metrics
                logger.info(f"Epoch {epoch}: Loss={epoch_loss:.4f}, Val Loss={val_loss:.4f}, "
                f"Acc={train_acc:.1f}%, Val Acc={val_acc:.1f}%"
                f"Time for this epoch={this_epoch_time:.2f}s, LR={current_lr:.6f}, Max Grad={max_grad:.4f}")
                      
        # Save final metrics history plot
        final_metrics_path = os.path.join(self.visualizer.exp_dir, "final_metrics_history.png")
        self.visualizer.plot_metrics_history(filepath=final_metrics_path)
        
        logger.info(f"Final metrics history saved to {final_metrics_path}")
        logger.info(f"Training completed in {len(epoch_times)} epochs with total time: {sum(epoch_times):.2f}s")
        
        return self.visualizer.history
                
    def validate(self, val_loader: DataLoader) -> float:
        """
        Validate the model using the validation set.
        The validation loop consists of the following steps:
        1. Iterate over the validation set in batches.
        2. Forward pass: Compute the predicted labels using the model.
        3. Compute the loss using the loss function.
        4. Return the average loss for the validation set.
        Args:
            val_loader (DataLoader): The DataLoader for the validation set.
        Returns:
            float: The average loss for the validation set.
        """
        
        val_loss = 0
        n_samples = 0
        for x_val, y_val in val_loader:
            y_val_pred = self.model.forward(x_val)
            loss = self.loss_fn.forward(y_val, y_val_pred)[0]
            val_loss += loss * x_val.shape[0]
            n_samples += x_val.shape[0]
        return val_loss / n_samples
    
    def compute_accuracy(self, loader: DataLoader) -> float:
        """
        Compute the accuracy of the model on the given DataLoader.
        The accuracy is defined as the number of correct predictions divided by the total number of predictions.
        Args:
            loader (DataLoader): The DataLoader for the dataset.
        Returns:
            float: The accuracy of the model on the dataset.
        """
        total_correct = 0
        total_samples = 0
        smooth_window = 5  
        running_acc = []
        
        for x_batch, y_batch in loader:
            pred = self.model.forward(x_batch)
            batch_correct = np.sum(np.argmax(pred, axis=1) == np.argmax(y_batch, axis=1))
            total_correct += batch_correct
            total_samples += len(y_batch)
            acc = 100 * batch_correct / len(y_batch)
            running_acc.append(acc)
            if len(running_acc) > smooth_window:
                running_acc.pop(0)
        
        return np.mean(running_acc) if running_acc else 0.0  # Return mean of the last few accuracies
    
    
    def train_with_cv(self, dataset: Dataset, cv: CrossValidator, plots_dir='kfold_plots', **kwargs) -> Dict[str, Union[float, List[float], Sequential, KFoldVisualizer]]:
        """
        Train with cross-validation and return the best model
        Args:
            dataset (Dataset): The dataset to use for training and validation.
            cv (CrossValidator): The cross-validator instance to use for splitting the dataset.
            plots_dir (str): Directory to save k-fold plots.
            **kwargs: Additional arguments for training (e.g., epochs, batch_size, lr_scheduler, etc.)
        Returns:
            dict: A dictionary containing the mean score, standard deviation, best model, fold scores,
                  and a KFoldVisualizer instance with aggregated results.
        """
        
        kfold_dir = os.path.join(self.plots_dir, 'kfold')
        os.makedirs(kfold_dir, exist_ok=True)
        
        cv_visualizer = KFoldVisualizer(len(cv.get_folds(dataset)), exp_dir=kfold_dir)
        fold_scores = []
        best_model = None
        best_score = float('inf')
 
                
        # Store initial model architecture
        initial_architecture = [(layer.__class__, layer.input_dim, layer.output_dim) 
                         if isinstance(layer, Linear) else layer.__class__ 
                          for layer in self.model.layers]
        
        # Create fresh learning rate scheduler for each fold with same parameters
        lr_params = {'initial_lr': kwargs.get('lr_scheduler').initial_lr, 
                     'warmup_epochs': kwargs.get('lr_scheduler').warmup_epochs}
       

        # Get the scheduler class and its parameters from the provided scheduler
        scheduler_class = type(kwargs['lr_scheduler'])
        scheduler_params = kwargs['lr_scheduler'].__dict__.copy()
        # Remove any stateful keys (like current_step) if present
        scheduler_params.pop('current_step', None)
        
        for fold_idx, (train_idx, val_idx) in enumerate(cv.get_folds(dataset)):
            logger.info(f"Training Fold {fold_idx + 1}")
        
            
            fold_dir = os.path.join(self.plots_dir, f'fold_{fold_idx + 1}')
            os.makedirs(fold_dir, exist_ok=True)
            
            # Reset visualization history for each fold
            self.visualizer = TrainingVisualizer(exp_dir=fold_dir)
            
            # Create fresh model with consistent initialization
            np.random.seed(42 + fold_idx)  # Consistent but different for each fold        
            layers = []
            for layer_info in initial_architecture:
                if isinstance(layer_info, tuple):
                    cls, in_dim, out_dim = layer_info
                    layer = cls(in_dim, out_dim)
                    # Use proper Xavier/Glorot initialization
                    layer.weights = np.random.randn(in_dim, out_dim) * np.sqrt(2.0 / in_dim)
                    layers.append(layer)
                else:
                    layers.append(layer_info())    
            self.model = Sequential(layers) 
            
            # Create fresh optimizer
            self.optimizer = SGD(
                learning_rate=lr_params['initial_lr'], 
                momentum=kwargs.get('momentum', 0.9),
                weight_decay=kwargs.get('weight_decay', 0.0)
            )

            # Create fresh learning rate scheduler
            lr_scheduler = scheduler_class(**scheduler_params)        
        
             # Create data loaders with consistent batch size
            train_loader = DataLoader(dataset, indices=train_idx, 
                                batch_size=kwargs.get('batch_size', 32),
                                shuffle=True)
            val_loader = DataLoader(dataset, indices=val_idx,
                              batch_size=kwargs.get('batch_size', 32))
        
        
            # Train this fold
        
            kwargs.pop('lr_scheduler', None)  # Remove scheduler from kwargs to avoid passing it to train method
            
            # Regular training but store history
            history = self.train(train_loader, val_loader, lr_scheduler=lr_scheduler, **kwargs)
            cv_visualizer.add_fold_history(history)
        
            # Compute fold score
            fold_score = self.validate(val_loader)
            fold_scores.append(fold_score)
        
            # Keep track of best model
            if fold_score < best_score:
                best_score = fold_score
                best_model = copy.deepcopy(self.model)
    
        cv_visualizer.plot_k_fold_results()
        
        # Restore best model
        self.model = best_model
    
        # Plot aggregated results
        cv_visualizer.plot_k_fold_results() 

    
        mean_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)
        logger.info(f"\nCross-validation score: {mean_score:.4f} ± {std_score:.4f}")
        return {
            'mean_score': mean_score,
            'std_score': std_score,
            'best_model': best_model,
            'fold_scores': fold_scores,
            'visualizer': cv_visualizer
        }
        
    def cleanup(self):
        """
        Cleanup method to close all plots and set the visualizer to None.
        """
        plt.close('all')
        self.visualizer =  None
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
# class TrainingVisualizer that stores the training history and plots the loss and accuracy

import matplotlib.pyplot as plt
import numpy as np
from typing import Optional
from matplotlib.gridspec import GridSpecFromSubplotSpec
from .layers import Linear, Sequential
from .optimizers import Optimizer
from .losses import Loss
from .data_utils import DataLoader


class TrainingVisualizer:
    '''
    TrainingVisualizer class to store the training history and plot the loss and accuracy.
    It provides methods to update the training history and plot the loss and accuracy.
    '''
    def __init__(self)->None:
        self.history = {
            'loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        self.grid = None
        self.grid_coords = None
        
    def update(self, loss: float, val_loss: float, train_acc: float, val_acc: float):
        '''
        Update the training history with the current loss and accuracy.
        Args:   
            loss (float): Current loss.
            val_loss (float): Current validation loss.
            train_acc (float): Current training accuracy.
            val_acc (float): Current validation accuracy.
        Returns:
            None
        '''
        self.history['loss'].append(loss)
        self.history['val_loss'].append(val_loss)
        self.history['train_acc'].append(train_acc)
        self.history['val_acc'].append(val_acc)

    def plot_metrics_history(self) -> None:
        '''
        Plot the training history.
        It plots the loss and accuracy for both training and validation sets.
        Returns:
            None
        '''
        assert len(self.history['loss']) > 0, "No training history to plot."
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        epochs = range(len(self.history['loss']))
        # Plot loss
        ax1.plot(epochs, self.history['loss'], 'b-', label='Train Loss')
        ax1.plot(epochs, self.history['val_loss'], 'r-', label='Val Loss')
        ax1.set_title('Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # Plot accuracy
        ax2.plot(epochs, self.history['train_acc'], 'b-', label='Train Accuracy')
        ax2.plot(epochs, self.history['val_acc'], 'r-', label='Val Accuracy')
        ax2.set_title('Accuracy')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()     
    
    def plot_decision_boundary(self, model: Sequential, x_train: np.ndarray, y_train: np.ndarray, ax: Optional[plt.Axes] = None) -> None:
        
        if x_train.shape[1] != 2:
            return
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        # Create grid first time only
        if self.grid is None:
            # Extend bounds a bit further for better visualization
            x_min, x_max = x_train[:, 0].min() - 1.0, x_train[:, 0].max() + 1.0
            y_min, y_max = x_train[:, 1].min() - 1.0, x_train[:, 1].max() + 1.0
        
            # Increase grid resolution for smoother boundaries
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                            np.linspace(y_min, y_max, 200))
            self.grid = np.c_[xx.ravel(), yy.ravel()]
            self.grid_coords = (xx, yy)
    
        # Get predictions for grid points
        grid_predictions = model.forward(self.grid)
        grid_predictions = np.argmax(grid_predictions, axis=1)
    
        # Plot decision boundary with better aesthetics
        ax.contourf(self.grid_coords[0], self.grid_coords[1], 
                 grid_predictions.reshape(self.grid_coords[0].shape),
                 alpha=0.15, cmap='viridis', levels=np.arange(4)-0.5)
    
        # Add contour lines to highlight boundaries
        ax.contour(self.grid_coords[0], self.grid_coords[1],
                grid_predictions.reshape(self.grid_coords[0].shape),
                colors='black', alpha=0.3, linewidths=0.5)
    
        # Plot training points with better visibility
        scatter = ax.scatter(x_train[:, 0], x_train[:, 1], 
                         c=np.argmax(y_train, axis=1), 
                         cmap='viridis',
                         edgecolors='white',
                         s=20,
                         alpha=0.6,  # Some transparency
                         linewidth=0.5)
    
        if hasattr(ax, 'figure'):
            ax.figure.colorbar(scatter, ax=ax, label='Class')
        
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.set_title('Decision Boundary', fontsize=14, pad=10)
    
        # Make plot more aesthetic
        ax.grid(True, alpha=0.2)
        plt.tight_layout()
        if ax is None:
            plt.show()
        
    def weights_gradients_heatmap(self, model: Sequential, optimizer: Optimizer, ax: Optional[plt.Axes]=None) -> None:
        '''
        Plot the weights and their updates during training.
        Args:
            model: Sequential model to visualize
            optimizer: Optimizer instance to calculate updates
            ax: Matplotlib axes to plot on, if None a new figure is created
        Returns:
            None
        '''
        
        # Get all Linear layers
        linear_layers = [(i, layer) for i, layer in enumerate(model.layers) 
                          if isinstance(layer, Linear)]
    
        if not linear_layers:
            print("No linear layers to visualize.")
            return
        
        num_layers = len(linear_layers)
    
        # Create figure with subplots for each layer
        if ax is None:
            fig = plt.figure(figsize=(12, 4 * num_layers + 1))  # Add extra space for title
            fig.patch.set_visible(False)
            gs = fig.add_gridspec(num_layers, 2, hspace=0.6, height_ratios=[1]*num_layers)
            axes = np.empty((num_layers, 2), dtype=object)
            for i in range(num_layers):
                axes[i, 0] = fig.add_subplot(gs[i, 0])
                axes[i, 1] = fig.add_subplot(gs[i, 1])
        else:
            fig = ax.get_figure()
            gs = GridSpecFromSubplotSpec(num_layers, 2, subplot_spec=ax.get_subplotspec(), hspace=0.6)
            axes = np.empty((num_layers, 2), dtype=object)
            for i in range(num_layers):
                axes[i, 0] = fig.add_subplot(gs[i, 0])
                axes[i, 1] = fig.add_subplot(gs[i, 1])
        
        for i, (layer_num, layer) in enumerate(linear_layers):
            # Left plot - weights
            weights_norm = layer.weights / np.abs(layer.weights).max()
            im1 = axes[i, 0].imshow(weights_norm, cmap='RdBu', vmin=-1, vmax=1)
            axes[i, 0].set_title(f'Layer {layer_num} Weights\nMax abs value: {np.abs(layer.weights).max():.4f}', 
                            pad=10)
            fig.colorbar(im1, ax=axes[i, 0])

            # Right plot - updates
            if layer.grad_weights is not None:
                update = optimizer.learning_rate * layer.grad_weights
                update_norm = update / np.abs(update).max() if np.abs(update).max() > 0 else update
                im2 = axes[i, 1].imshow(update_norm, cmap='RdBu', vmin=-1, vmax=1)
                axes[i, 1].set_title(f'Layer {layer_num} Updates (lr={optimizer.learning_rate:.6f})\nMax abs value: {np.abs(update).max():.4f}', 
                                pad=10)
                fig.colorbar(im2, ax=axes[i, 1])

            # Add labels with consistent spacing
            for ax in [axes[i, 0], axes[i, 1]]:
                ax.set_xlabel('Output features', labelpad=10)
                ax.set_ylabel('Input features', labelpad=10)
            
            # Adjust tick labels if needed
            for ax in [axes[i, 0], axes[i, 1]]:
                ax.tick_params(axis='both', which='major', labelsize=8)

        plt.suptitle('Weight Values and Their Updates', y=1.02, fontsize=14)
    
        if ax is None:
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to prevent overlap
            plt.show()
        
    
    
        
    def plot_loss_landscape(self, model: Sequential, loader: DataLoader, loss_fn: Loss, ax: Optional[plt.Axes]=None)->None:
        """Visualize loss landscape around current weights"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        losses = []
        epsilons = np.linspace(-1, 1, 20)
    
        # Store original weights
        original_weights = [(layer.weights.copy(), layer.bias.copy()) 
                       for layer in model.layers if isinstance(layer, Linear)]
    
        for eps in epsilons:
            # Perturb weights
            for layer, (w, b) in zip([l for l in model.layers if isinstance(l, Linear)], 
                                original_weights):
                layer.weights = w + eps * np.random.randn(*w.shape) * 0.1
                layer.bias = b + eps * np.random.randn(*b.shape) * 0.1
        
            # Compute loss
            total_loss = 0
            n_samples = 0
            for x_batch, y_batch in loader:
                y_pred = model.forward(x_batch)
                loss, _ = loss_fn.forward(y_batch, y_pred)
                total_loss += loss * len(x_batch)
                n_samples += len(x_batch)
            losses.append(total_loss / n_samples)
    
            # Restore original weights
            for layer, (w, b) in zip([l for l in model.layers if isinstance(l, Linear)], 
                            original_weights):
                layer.weights = w
                layer.bias = b
    
        ax.plot(epsilons, losses)
        ax.set_xlabel('Perturbation magnitude')
        ax.set_ylabel('Loss')
        ax.set_title('Loss Landscape')
        ax.grid(True)
        if ax is None:
            plt.show()
        

class KFoldVisualizer(TrainingVisualizer):
    """Extended visualizer for k-fold cross validation"""
    def __init__(self, k_folds: int) -> None:
        super().__init__()
        self.k_folds = k_folds
        self.fold_histories = []
        
    def add_fold_history(self, fold_history: dict) ->None:
        """Store history for one fold"""
        self.fold_histories.append(fold_history)
    
    def plot_k_fold_results(self)-> None:
        """Plot aggregated results across folds"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot individual fold histories
        for i, hist in enumerate(self.fold_histories):
            ax1.plot(hist['val_loss'], alpha=0.3, label=f'Fold {i+1}')
            ax2.plot(hist['val_acc'], alpha=0.3)
        
        # Plot mean ± std
        val_losses = np.array([h['val_loss'] for h in self.fold_histories])
        val_accs = np.array([h['val_acc'] for h in self.fold_histories])
        
        epochs = range(len(val_losses[0]))
        mean_loss = np.mean(val_losses, axis=0)
        std_loss = np.std(val_losses, axis=0)
        mean_acc = np.mean(val_accs, axis=0)
        std_acc = np.std(val_accs, axis=0)
        
        ax1.plot(epochs, mean_loss, 'r-', label='Mean Loss', linewidth=2)
        ax1.fill_between(epochs, mean_loss-std_loss, mean_loss+std_loss, alpha=0.2)
        ax1.set_title(f'Validation Loss Across {self.k_folds} Folds')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        ax2.plot(epochs, mean_acc, 'r-', label='Mean Accuracy', linewidth=2)
        ax2.fill_between(epochs, mean_acc-std_acc, mean_acc+std_acc, alpha=0.2)
        ax2.set_title(f'Validation Accuracy Across {self.k_folds} Folds')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
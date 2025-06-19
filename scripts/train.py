# script that takes all the necessary arguments from the user and trains the model


import argparse
import os
import time
import numpy as np
from src.data_utils import *
from src.layers import Sequential, Linear, ReLU
from src.losses import CrossEntropySoftMax
from src.optimizers import SGD 
from src.schedulers import WarmupLRScheduler, ExponentialLRScheduler, CosineAnnealingLRScheduler
from src.trainer import Trainer
from src.cross_validator import CrossValidator
from src.utils import one_hot_encode


def setup_experiment_dir(experiment_name: str) -> str:
    """
    Set up clean experiment directory with sync
    """
    exp_dir = os.path.join('experiments', experiment_name)
    if os.path.exists(exp_dir):
        print(f"Removing existing experiment directory: {exp_dir}")
        for root, dirs, files in os.walk(exp_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(exp_dir)
        # Wait for filesystem to catch up
        time.sleep(0.1)
        
    os.makedirs(exp_dir)
    os.sync()
    return exp_dir

def main():
    
    #parse command line arguments
    parser = argparse.ArgumentParser(description="Train a neural network model.")
    parser.add_argument("--dataset", choices=["mnist", "digits", "synthetic"], default="mnist",  help='Dataset to use (mnist, digits, or synthetic: spirals)')
    parser.add_argument("--n-samples", type=int, default=1000,
                    help="Number of total points when using synthetic data")
    parser.add_argument("--n-classes", type=int, default=2,
                    help="Number of classes for synthetic data")
    parser.add_argument("--class-sep", type=float, default=1.0,
                    help="Cluster separation (for make_classification or similar)")
    parser.add_argument('--data-dir', type=str, default='data', help='Directory to save/load dataset')
    parser.add_argument('--n-folds', type=int, default=5, help='Number of folds for cross-validation')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate for the optimizer')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for the optimizer')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay (L2 regularization)')
    parser.add_argument('--scheduler', type=str, default='warmup', choices=['none', 'warmup','cosine'], help='Learning rate scheduler type')
    parser.add_argument('--warmup-epochs', type=int, default=5, help='Number of warmup epochs for the scheduler')
    parser.add_argument('--hidden-dims', type=str, default='128',
                       help='Comma-separated list of hidden layer dimensions (e.g., 256,128,64)')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--experiment-name', type=str, default='demo_run',
                       help='Name for the experiment (used for output organization)')
    parser.add_argument('--log-interval', type=int, default=100, help='Interval for logging training progress')
    args = parser.parse_args()


    if args.seed is not None:
        np.random.seed(args.seed)
        print(f"Using seed: {args.seed}")

    data_dir = args.data_dir if args.data_dir else './data'

    # download / generate dataset depending on many options
    if args.dataset == 'synthetic':
            # Generate synthetic spiral dataset
            print("Generating spiral dataset...")
            X, y = generate_spiral_data(args.n_samples, args.n_classes, args.class_sep, args.seed)    
    else:
        # Load dataset from data_utils
        print(f"Loading {args.dataset} dataset...")
        X, y = load_openml_dataset(args.dataset, data_dir)

    #convert label to one-hot encoding
    print("Converting labels to one-hot encoding...")
    y = one_hot_encode(y)
    
    # divide dataset into train and test sets
    print("Splitting dataset into train and test sets...")
    train_dataset, test_dataset = DataLoader.holdout_split(Dataset(X, y), test_size=0.2, batch_size=args.batch_size)
    print(f"Train set size: {len(train_dataset)}, Test set size: {len(test_dataset)}")
    
    # Create model, loss function, optimizer, and trainer
    
    input_dim = X.shape[1]
    hidden_dims  = [int(dim) for dim in args.hidden_dims.split(',')] or [128]  # Default to 128 if not specified
    num_classes = y.shape[1]
    
    #print(f"Input dimension: {input_dim}, Hidden dimension: {hidden_dim}, Number of classes: {num_classes}")
    
    # Create model architecture dynamically
    initial_architecture = []
    prev_dim = input_dim
    
    for hidden_dim in hidden_dims:
        initial_architecture.extend([
            (Linear, prev_dim, hidden_dim),
            ReLU
        ])
        prev_dim = hidden_dim
        
    # Add final layer
    initial_architecture.append((Linear, prev_dim, num_classes))
    
    # Create layers from architecture specification
    layers = []
    for layer_info in initial_architecture:
        if isinstance(layer_info, tuple):
            cls, in_dim, out_dim = layer_info
            layers.append(cls(in_dim, out_dim))
        else:
            layers.append(layer_info())

    # Initialize model with list of layers
    model = Sequential(layers)  # Pass the list directly, not unpacked
    
    print("Model architecture:")
    print(model.summary())
    
    exp_dir = setup_experiment_dir(args.experiment_name)
    
    # save model architecture to a file
    #create experiment directory if it doesn't exist
    os.makedirs(exp_dir, exist_ok=True)
    arch_file = os.path.join(exp_dir, "model_architecture.txt")
        
    model.save_architecture(arch_file)
    print("Model architecture saved to model_architecture.txt")
    # save experiment parameters to a file
    with open(os.path.join(exp_dir, "experiment_params.txt"), "w") as f:
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Number of samples: {args.n_samples}\n")
        f.write(f"Number of classes: {args.n_classes}\n")
        f.write(f"Class separation: {args.class_sep}\n")
        f.write(f"Data directory: {args.data_dir}\n")
        f.write(f"Number of folds: {args.n_folds}\n")
        f.write(f"Epochs: {args.epochs}\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"Learning rate: {args.lr}\n")
        f.write(f"Momentum: {args.momentum}\n")
        f.write(f"Weight decay: {args.weight_decay}\n")
        f.write(f"Scheduler: {args.scheduler}\n")
        f.write(f"Warmup epochs: {args.warmup_epochs}\n")
        f.write(f"Experiment name: {args.experiment_name}\n")
        f.write(f"Hidden dimensions: {hidden_dims}\n")
        f.write(f"log interval: {args.log_interval}\n")
        if args.seed is not None:
            f.write(f"Seed: {args.seed}\n")
    print("Experiment parameters saved to experiment_params.txt")

    loss_fn = CrossEntropySoftMax()
    optimizer = SGD(learning_rate=args.lr)

    if args.scheduler == "warmup":
        lr_scheduler = WarmupLRScheduler(initial_lr=args.lr,
                                     warmup_epochs=args.warmup_epochs)
    elif args.scheduler == "exp":
        lr_scheduler = ExponentialLRScheduler(initial_lr=args.lr,
                                          gamma=args.gamma)
    elif args.scheduler == "cosine":
        lr_scheduler = CosineAnnealingLRScheduler(initial_lr=args.lr,
                                                  T_max=args.epochs,
                                                  eta_min=1e-6,
                                                  warmup_epochs=args.warmup_epochs,
                                                  warmup_start_lr=0.0001)
    else:
        lr_scheduler = None
        
    trainer = Trainer(model=model,
                      loss_fn=loss_fn,
                      optimizer=optimizer,
                      exp_dir=exp_dir
                      )


    trainer.train_with_cv(
        train_dataset, 
        cv=CrossValidator(k=args.n_folds, random_state=args.seed),
        debug=False,
        epochs=args.epochs,
        patience=150,
        batch_size=args.batch_size,
        show_plots_logs=True,
        lr_scheduler=lr_scheduler,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        log_interval=args.log_interval,
        )

    # Create test loader and evaluate
    test_acc = trainer.compute_accuracy(DataLoader(test_dataset))
    print(f"\nFinal Test Accuracy: {test_acc:.2f}%")

if __name__ == "__main__":
    main()
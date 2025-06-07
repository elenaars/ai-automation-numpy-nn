# script that takes all the necessary arguments from the user and trains the model

# IN PROGRESS

# Sample usage: 
#python scripts/train.py \
#  --dataset mnist \
#  --data-dir data/mnist \
#  --epochs 20 \
#  --batch-size 64 \
#  --lr 0.01 \
#  --momentum 0.9 \
#  --weight-decay 1e-4 \
#  --scheduler warmup \
#  --warmup-epochs 5 \
#  --experiment-name demo_run



import argparse
import os
import numpy as np
from src.data_utils import *
from src.layers import Sequential, Linear, ReLU
from src.losses import CrossEntropySoftMax
from src.optimizers import SGD 
from src.schedulers import WarmupLRScheduler, ExponentialLRScheduler
from src.trainer import Trainer
from src.cross_validator import CrossValidator
from src.utils import one_hot_encode


def main():
    
    #parse command line arguments
    parser = argparse.ArgumentParser(description="Train a neural network model.")
    parser.add_argument("--dataset", choices=["mnist", "digits", "synthetic"], default="mnist",  help='Dataset to use (mnist, digits, or synthetic: spirals)')
    parser.add_argument("--n‐samples", type=int, default=1000,
                    help="Number of total points when using synthetic data")
    parser.add_argument("--n‐classes", type=int, default=2,
                    help="Number of classes for synthetic data")
    parser.add_argument("--class‐sep", type=float, default=1.0,
                    help="Cluster separation (for make_classification or similar)")
    parser.add_argument('--data-dir', type=str, default='data', help='Directory to save/load dataset')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate for the optimizer')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for the optimizer')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay (L2 regularization)')
    parser.add_argument('--scheduler', type=str, default='warmup', choices=['none', 'warmup'], help='Learning rate scheduler type')
    parser.add_argument('--warmup-epochs', type=int, default=5, help='Number of warmup epochs for the scheduler')
    parser.add_argument('--experiment-name', type=str, default='demo_run', help='Name of the experiment for logging')
    parser.add_argument('--hidden-dim', type=int, default=128, help='Hidden dimension for the model')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    args = parser.parse_args()


    if args.seed is not None:
        np.random.seed(args.seed)
        print(f"Using seed: {args.seed}")

    # download / generate dataset depending on many options
    match args.dataset:
        case 'synthetic':
            # Generate synthetic spiral dataset
            print("Generating spiral dataset...")
            X, y = generate_spiral_data(args.n_samples, args.n_classes, args.class_sep, args.seed)    
        case 'mnist':
            # Load MNIST dataset
            print("Loading MNIST dataset...")
            X, y = download_mnist_data()
        case 'digits':
            # Load Digits dataset
            print("Loading Digits dataset...")
            X, y = download_digits_data()
        case 'fashion_mnist':
            # Load Fashion MNIST dataset
            print("Loading Fashion MNIST dataset...")
            X, y = download_fashion_mnist_data()
        case 'iris':
            # Load Iris dataset
            print("Loading Iris dataset...")
            X, y = download_iris_data()
        case _:
            raise ValueError(f"Unknown dataset: {args.dataset}")

    #converl label to one-hot encoding
    print("Converting labels to one-hot encoding...")
    y = one_hot_encode(y, num_classes=5) if args.dataset == 'synthetic' else one_hot_encode(y)
    # divide dataset into train and test sets
    print("Splitting dataset into train and test sets...")
    train_dataset, test_dataset = DataLoader.holdout_split(Dataset(X, y), test_size=0.2, batch_size=args.batch_size)
    print(f"Train set size: {len(train_dataset)}, Test set size: {len(test_dataset)}")
    
    # Create model, loss function, optimizer, and trainer
    
    input_dim = X.shape[1]
    hidden_dim  = args.hidden_dim or 128
    num_classes = y.shape[1]
    
    print(f"Input dimension: {input_dim}, Hidden dimension: {hidden_dim}, Number of classes: {num_classes}")
    
    model = Sequential([
    Linear(input_dim,  hidden_dim),
    ReLU(),
    Linear(hidden_dim, num_classes),
    ])
    
    print("Model architecture:")
    print(model.summary())

    loss_fn = CrossEntropySoftMax()
    optimizer = SGD(learning_rate=args.lr)

    if args.scheduler == "warmup":
        lr_scheduler = WarmupLRScheduler(initial_lr=args.lr,
                                     warmup_epochs=args.warmup_epochs)
    elif args.scheduler == "exp":
        lr_scheduler = ExponentialLRScheduler(initial_lr=args.lr,
                                          gamma=args.gamma)
    else:
        lr_scheduler = None
        
    trainer = Trainer(model=model,
                      loss_fn=loss_fn,
                      optimizer=optimizer,
                      #experiment_name=args.experiment_name,
                      #log_dir=os.path.join(args.data_dir, "logs"),
                      #save_dir=os.path.join(args.data_dir, "models"),
                      #seed=args.seed
                      )


    trainer.train_with_cv(
        train_dataset, 
        cv=CrossValidator("k-fold", k=5),
        debug=False,
        epochs=args.epochs,
        patience=150,
        log_interval=100,
        batch_size=args.batch_size,
        show_plots_logs=True,
        lr_scheduler=lr_scheduler,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        )

    # Create test loader and evaluate
    test_acc = trainer.compute_accuracy(test_dataset)
    print(f"\nFinal Test Accuracy: {test_acc:.2f}%")

if __name__ == "__main__":
    main()
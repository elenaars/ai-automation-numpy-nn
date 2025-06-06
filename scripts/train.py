# script that takes all the necessary arguments from the user and trains the model

# IN PROGRESS


import argparse
import os
import numpy as np
from src.utils import generate_spiral_data
from src.data_utils import Dataset, DataLoader
from src.layers import Sequential, Linear, ReLU
from src.losses import CrossEntropySoftMax
from src.optimizers import SGD, WarmupLRScheduler
from src.trainer import Trainer
from src.cross_validator import CrossValidator


x_train, y_train = generate_spiral_data(1000, 5)
x_test, y_test = generate_spiral_data(200, 5)
train_dataset = Dataset(x_train, y_train)
test_dataset = Dataset(x_test, y_test)

model = Sequential([
    Linear(2,64),
    ReLU(),
    Linear(64, 32),
    ReLU(),
    Linear(32, 5),
])

loss_fn = CrossEntropySoftMax()
optimizer = SGD(learning_rate=0.1)
trainer = Trainer(model, loss_fn, optimizer)


trainer.train_with_cv(
    train_dataset, 
    cv=CrossValidator("k-fold", k=5),
    debug=False,
    epochs=2000,
    patience=150,
    log_interval=100,
    batch_size=32,
    show_plots_logs=True,
    lr_scheduler= WarmupLRScheduler(
        initial_lr=0.01, 
        warmup_epochs=100,
    ),
    weight_decay=0.001,
    momentum=0.9,
    )

# Create test loader and evaluate
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)
test_acc = trainer.compute_accuracy(test_loader)
print(f"\nFinal Test Accuracy: {test_acc:.2f}%")
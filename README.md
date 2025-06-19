# ai-automation-numpy-nn
From-scratch NumPy neural-network training pipeline with end-to-end automation


# NumPy Neural Network Implementation

A clean implementation of neural networks using only NumPy, featuring:
- Custom layers (Linear, ReLU)
- Various optimizers and learning rate schedulers
- Cross-validation support
- Training visualization
- Support for MNIST, Digits and synthetic datasets

## Installation
```bash
pip install -r requirements.txt
```

## Usage
Basic example:
```bash
python -m scripts.train \
    --dataset mnist \
    --hidden-dims 128,64 \
    --epochs 50 \
    --batch-size 32 \
    --lr 0.01 \
    --scheduler cosine
    --log-interval 10
```

## Project Structure
- `src/`: Core implementation
  - `layers.py`: Neural network layers
  - `optimizers.py`: SGD optimizer
  - `schedulers.py`: Learning rate schedulers
  - `losses.py`: Loss functions
  - `trainer.py`: Training logic
  - `visualizer.py`: Training visualization
  - `data_utils.py`: Dataset utilities
  - `cross_validator.py`: K-fold cross validation
- `scripts/`: Training scripts
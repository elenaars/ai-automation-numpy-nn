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
Basic examples:
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

```bash
python -m scripts.train \
 --dataset synthetic \
 --n-samples 2000 \
 --n-classes 3  \
 --class-sep 1.5 \
 --lr 0.01 \
 --hidden-dims 256,128,64 \
 --experiment-name spiral_3 \
 --epochs 1000 \
 --batch-size 32 \
 --momentum 0.9 \
 --weight-decay 1e-5 \
 --scheduler cosine \
 --warmup-epochs 20 \
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
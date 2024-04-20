# Tuning and Training MLP Models

This module is based on the [pytorch github](https://github.com/pytorch/vision/tree/main/references/classification) references for image classification, which is subject to the BSD 3-Clause License at the top of the respective files. It is used to train simple MLP models for image classification tasks. Note that for this module to work properly you must first register the model and dataset as described in the [Support for Model Architectures](https://gitlab.aws.dev/adavidho/mlp-rank-pruning/-/blob/main/mlprank/architectures/README.md) documentation.


## Tuning Learning Rate and Weight Decay

Most hyperparameters are set to recommended defaults. However, as weight decay and learning rate are heavily model dependent, they are tuned for each network using the `tuning_grid.py` script.

### Examples:

**TwoLayerMLP:** 
Tuning time = 0:22:32, Learning Rate = 0.005, Weight Decay = 1e-08.
```sh
torchrun --nproc_per_node=8 training/tuning_grid.py \
    --model-name TwoLayerMLP \
    --data-name MNIST \
    --epochs 20 \
    --lr-grid 0.05 0.01 0.005 0.001 0.0005 0.0001 \
    --wd-grid 0.00001 0.000001 0.0000001 0.00000001 0.000000001 0.0000000001
```

**ThreeLayerMLP:**
Tuning time = 0:57:45, Learning Rate = 0.005, Weight Decay = 1e-10.
```sh
torchrun --nproc_per_node=8 training/tuning_grid.py \
    --model-name ThreeLayerMLP \
    --data-name FashionMNIST \
    --epochs 50 \
    --lr-grid 0.05 0.01 0.005 0.001 0.0005 0.0001 \
    --wd-grid 0.00001 0.000001 0.0000001 0.00000001 0.000000001 0.0000000001
```

**SixLayerMLP:**
Tuning time = 2:31:02, Learning Rate = 0.0001, Weight Decay = 1e-09.
```sh
torchrun --nproc_per_node=8 training/tuning_grid.py \
    --model-name SixLayerMLP \
    --data-name CIFAR10 \
    --epochs 100 \
    --lr-grid 0.05 0.01 0.005 0.001 0.0005 0.0001 \
    --wd-grid 0.00001 0.000001 0.0000001 0.00000001 0.000000001 0.0000000001
```

**TwelveLayerMLP:**
Tuning time = 5:44:05, Learning Rate = 0.0001, Weight Decay = 1e-09.
```sh
torchrun --nproc_per_node=8 training/tuning_grid.py \
    --model-name TwelveLayerMLP \
    --data-name CIFAR100 \
    --epochs 200 \
    --lr-grid 0.05 0.01 0.005 0.001 0.0005 0.0001 \
    --wd-grid 00.00001 0.000001 0.0000001 0.00000001 0.000000001 0.0000000001
```


## Training MLP Moldes on Image Data

One can also use the `tuning_grid.py` script for training by simply passing a single value to both `lr_grid` and `wd_grid`.  Training is logged with wandb by default.

### Examples:

**TwoLayerMLP:**
Size = 200k params, Acc1 = 100, Acc5 = 100, ValAcc1 = 98.233, ValAcc5 = 100.
```sh
torchrun --nproc_per_node=8 training/tuning_grid.py \
    --model-name TwoLayerMLP \
    --data-name MNIST \
    --output-dir artifacts/models/TwoLayerMLP/ \
    --epochs 20 \
    --lr-grid 0.005 \
    --wd-grid 0.00000001
```

**ThreeLayerMLP:**
Size = 600k params, Acc1 = 98.438, Acc5 = 100, ValAcc1 = 98.233, ValAcc5 = 100.
```sh
torchrun --nproc_per_node=8 training/tuning_grid.py \
    --model-name ThreeLayerMLP \
    --data-name FashionMNIST \
    --output-dir artifacts/models/ThreeLayerMLP/ \
    --epochs 100 \
    --lr-grid 0.005 \
    --wd-grid 0.0000000001
```

**SixLayerMLP:**
Size = 2m params, Acc1 = 64.063, Acc5 = 93.75, ValAcc1 = 58.85, ValAcc5 = 93.26.
```sh
torchrun --nproc_per_node=8 training/tuning_grid.py \
    --model-name SixLayerMLP \
    --data-name CIFAR10 \
    --output-dir artifacts/models/SixLayerMLP/ \
    --epochs 200 \
    --lr-grid 0.0001 \
    --wd-grid 0.000000001
```

**TwelveLayerMLP:**
Size = 16m params, Acc1 = 48.438, Acc5 = 70.313, ValAcc1 = 31.52, ValAcc5 = 56.91.
```sh
torchrun --nproc_per_node=8 training/tuning_grid.py \
    --model-name TwelveLayerMLP \
    --data-name CIFAR100 \
    --output-dir artifacts/models/TwelveLayerMLP/ \
    --epochs 200 \
    --lr-grid 0.0001 \
    --wd-grid 0.000000001
```

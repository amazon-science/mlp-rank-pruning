# BSD 3-Clause License

# Copyright (c) Soumith Chintala 2016,
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.

# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import datetime
import logging as log
import os
import time
import warnings
import wandb
import json
import argparse

from training import presets
import torch
import torch.utils.data
import torchvision
import torchvision.transforms
from training import utils
from training.sampler import RASampler
from torch import nn
from torch.utils.data.dataloader import default_collate
from torchvision.transforms.functional import InterpolationMode
from training.transforms import get_mixup_cutmix
from mlprank import registry
from mlprank.pipeline import get_dataset
from torchvision.transforms import ToTensor
from torchvision.transforms import AutoAugmentPolicy, InterpolationMode
import torchvision.transforms as transforms

LOG_LEVEL = log.INFO
LOG_HANDLE = os.path.basename(__file__)[:-3]

# Initialise logging
log.basicConfig()
log.root.setLevel(LOG_LEVEL)
tune_log = log.getLogger(LOG_HANDLE)


def main(args):
    """
    Perform hyperparameter tuning for a given model using grid search.

    Args:
        args (argparse.Namespace): Command-line arguments and
            configurations.

    Returns:
        None

    This function performs hyperparameter tuning for a given model using grid
    search. It iterates over the provided learning rate and weight decay grid
    and trains the model with each combination of hyperparameters. For each
    combination, it records the training and testing accuracies and logs them
    using Weights & Biases. If only a single training configuration is
    provided, it skips logging to Weights & Biases and directly saves the
    results to a JSON file. After tuning is completed, it logs the total
    tuning time and indicates the completion of the tuning process.
    """

    grid_results = {}
    utils.init_distributed_mode(args)
    start_tuning_time = time.time()
    if len(args.lr_grid) == 1 and len(args.wd_grid) == 1:
        single_training = True
    else:
        single_training = False
    for lr in args.lr_grid:
        for weight_decay in args.wd_grid:
            tune_log(f"Train for lr={lr} and wd={weight_decay}")
            args.weight_decay = weight_decay
            args.lr = lr
            train_acc1, train_acc5, test_acc1, test_acc5 = train_one_model(
                args, single_training
            )
            grid_results[str((lr, weight_decay))] = {
                "train_acc1": train_acc1,
                "train_acc5": train_acc5,
                "test_acc1": test_acc1,
                "test_acc5": test_acc5,
            }
            if not single_training and args.wandb:
                wandb.init(
                    # Set the project where this run will be logged
                    project="MLP-Tuning",
                    group=f"{args.model_name}-lr{lr}-wd{weight_decay}",
                    # Track hyperparameters and run metadata
                    config={
                        "lr": lr,
                        "weight_decay": weight_decay,
                        "train_acc1": train_acc1,
                        "train_acc5": train_acc5,
                        "test_acc1": test_acc1,
                        "test_acc5": test_acc5,
                        "args": args,
                    },
                )
                wandb.log(
                    {
                        "train_acc1": train_acc1,
                        "train_acc5": train_acc5,
                        "test_acc1": test_acc1,
                        "test_acc5": test_acc5,
                    }
                )
                wandb.finish()
    if not single_training:
        with open(
            f"artifacts/tuning/{args.model_name}-grid.json", "w"
        ) as outfile:
            json.dump(grid_results, outfile)
    total_tuning_time = time.time() - start_tuning_time
    total_tuning_time_str = str(
        datetime.timedelta(seconds=int(total_tuning_time))
    )
    tune_log(f"Tuning time: {total_tuning_time_str}")
    tune_log("Done!!!")


def train_one_epoch(
    model,
    criterion,
    optimizer,
    data_loader,
    device,
    epoch,
    args,
    model_ema=None,
    scaler=None,
    single_training=False,
    wandb_log=False,
):
    """
    Trains the neural network model for one epoch and returns the accuracy.

    Args:
        model (torch.nn.Module): The neural network model to train.
        criterion (torch.nn.Module): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer for updating model
            parameters.
        data_loader (torch.utils.data.DataLoader): The data loader for
            training data.
        device (torch.device): The device (CPU or GPU) to perform training.
        epoch (int): The current epoch number.
        args (argparse.Namespace): Command-line arguments and configurations.
        model_ema (Optional[torch.nn.Module]): The exponential moving average
            of model parameters.
        scaler (Optional[torch.cuda.amp.GradScaler]): The gradient scaler for
            mixed precision training.
        single_training (bool): Whether to perform single training without
            logging.

    Returns:
        Tuple[float, float]: The top-1 and top-5 accuracies achieved
        during training.
    """
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        "lr", utils.SmoothedValue(window_size=1, fmt="{value}")
    )
    metric_logger.add_meter(
        "img/s", utils.SmoothedValue(window_size=10, fmt="{value}")
    )

    header = f"Epoch: [{epoch}]"
    for i, (image, target) in enumerate(
        metric_logger.log_every(data_loader, args.print_freq, header)
    ):
        start_time = time.time()
        image, target = image.to(device), target.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)
            loss = criterion(output, target)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            if args.clip_grad_norm is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    model.parameters(), args.clip_grad_norm
                )
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.clip_grad_norm is not None:
                nn.utils.clip_grad_norm_(
                    model.parameters(), args.clip_grad_norm
                )
            optimizer.step()

        if model_ema and i % args.model_ema_steps == 0:
            model_ema.update_parameters(model)
            if epoch < args.lr_warmup_epochs:
                model_ema.n_averaged.fill_(0)

        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        batch_size = image.shape[0]
        metric_logger.update(
            loss=loss.item(), lr=optimizer.param_groups[0]["lr"]
        )
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
        metric_logger.meters["img/s"].update(
            batch_size / (time.time() - start_time)
        )
        if i % 100 == 0 and single_training and wandb_log:
            wandb.log(
                {"acc1": acc1.item(), "acc5": acc5.item(), "loss": loss.item()}
            )
    return acc1.item(), acc5.item()


def evaluate(
    model,
    criterion,
    data_loader,
    device,
    print_freq=100,
    log_suffix="",
    single_training=False,
    wandb_log=False,
):
    """
    Evaluates the neural network model.

    Args:
        model (torch.nn.Module): The neural network model to evaluate.
        criterion (torch.nn.Module): The loss function.
        data_loader (torch.utils.data.DataLoader): The data loader for
            evaluation data.
        device (torch.device): The device (CPU or GPU) to perform evaluation.
        print_freq (int, optional): The frequency of logging during
            evaluation.
        log_suffix (str, optional): The suffix to add to the evaluation log
            header.
        single_training (bool, optional): Whether to perform single training
            without logging.

    Returns:
        Tuple[float, float]: The top-1 and top-5 accuracies achieved during
        evaluation.
    """
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f"Test: {log_suffix}"

    num_processed_samples = 0
    with torch.inference_mode():
        for image, target in metric_logger.log_every(
            data_loader, print_freq, header
        ):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(image)
            loss = criterion(output, target)

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
            num_processed_samples += batch_size
    # gather the stats from all processes

    num_processed_samples = utils.reduce_across_processes(
        num_processed_samples
    )
    if (
        hasattr(data_loader.dataset, "__len__")
        and len(data_loader.dataset) != num_processed_samples
        and torch.distributed.get_rank() == 0
    ):
        # See FIXME above
        warnings.warn(
            f"It looks like the dataset has {len(data_loader.dataset)} "
            f"samples, but {num_processed_samples} samples were used for the "
            "validation, which might bias the results. "
            "Try adjusting the batch size and / or the world size. "
            "Setting the world size to 1 is always a safe bet."
        )

    metric_logger.synchronize_between_processes()
    tune_log(
        (
            f"{header} Acc@1 {metric_logger.acc1.global_avg:.3f}"
            f"Acc@5 {metric_logger.acc5.global_avg:.3f}"
        )
    )
    if single_training and wandb_log:
        wandb.log(
            {
                "val_acc1": metric_logger.acc1.global_avg,
                "val_acc5": metric_logger.acc5.global_avg,
            }
        )
    return metric_logger.acc1.global_avg, metric_logger.acc5.global_avg


def _get_cache_path(filepath):
    """
    Generates the cache path for the given file path using SHA-1 hashing.

    Args:
        filepath (str): The path to the file for which the cache path is
        generated.

    Returns:
        str: The cache path.
    """
    import hashlib

    h = hashlib.sha1(filepath.encode()).hexdigest()
    cache_path = os.path.join(
        "~", ".torch", "vision", "datasets", "imagefolder", h[:10] + ".pt"
    )
    cache_path = os.path.expanduser(cache_path)
    return cache_path


def load_data(dataset_name) -> tuple:
    """
    Load one of the datasets registered with the mlprank module.

    Args:
        dataset_name (str): The name of the dataset to load.

    Returns:
        tuple: A tuple containing the train and test datasets along with their
        samplers.
    """
    if dataset_name in ["CIFAR10", "CIFAR100"]:
        transform = transforms.Compose(
            [
                transforms.AutoAugment(
                    AutoAugmentPolicy.CIFAR10,
                    interpolation=InterpolationMode.NEAREST,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    else:
        transform = ToTensor()
    wrapper = registry.DATASETS[dataset_name]["wrapper"]
    path = registry.DATASETS[dataset_name]["path"]
    # Load data
    data = wrapper(root=path, train=True, download=True, transform=transform)
    train_size = 0.8
    dev_size = 1 - train_size
    train_data, test_data = torch.utils.data.random_split(
        data, [train_size, dev_size]
    )
    train_data.classes = data.classes
    test_data.classes = data.classes
    if args.distributed:
        if hasattr(args, "ra_sampler") and args.ra_sampler:
            train_sampler = RASampler(
                train_data, shuffle=True, repetitions=args.ra_reps
            )
        else:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_data
            )
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            test_data, shuffle=False
        )
    else:
        train_sampler = torch.utils.data.RandomSampler(train_data)
        test_sampler = torch.utils.data.SequentialSampler(test_data)
    return train_data, test_data, train_sampler, test_sampler


def train_one_model(args, single_training):
    """
    Train a model using the provided arguments.

    Args:
        args (argparse.Namespace): Command-line arguments.
        single_training (bool): Whether it's a single training session.

    Returns:
        tuple: A tuple containing training and testing accuracies.
    """
    if single_training and args.wandb:
        experiment_name = f"Training-{args.model_name}-{args.data_name}"
        run = wandb.init(
            # Set the project where this run will be logged
            project="MLP-Training",
            group=experiment_name,
            name=experiment_name.lower(),
            # Track hyperparameters and run metadata
            config=vars(args),
        )
    if args.output_dir:
        utils.mkdir(args.output_dir)

    device = torch.device(args.device)

    if args.use_deterministic_algorithms:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True

    train_dir = os.path.join(args.data_path, "train")
    val_dir = os.path.join(args.data_path, "val")
    dataset, dataset_test, train_sampler, test_sampler = load_data(
        args.data_name
    )
    num_classes = len(dataset.classes)
    mixup_cutmix = get_mixup_cutmix(
        mixup_alpha=args.mixup_alpha,
        cutmix_alpha=args.cutmix_alpha,
        num_classes=num_classes,
        use_v2=args.use_v2,
    )
    if mixup_cutmix is not None:

        def collate_fn(batch):
            return mixup_cutmix(*default_collate(batch))

    else:
        collate_fn = default_collate

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        sampler=test_sampler,
        num_workers=args.workers,
        pin_memory=True,
    )

    tune_log("Creating model")
    model = registry.MLP_MODEL[args.model_name]["wrapper"]()
    model.to(device)

    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    custom_keys_weight_decay = []
    if args.bias_weight_decay is not None:
        custom_keys_weight_decay.append(("bias", args.bias_weight_decay))
    if args.transformer_embedding_decay is not None:
        for key in [
            "class_token",
            "position_embedding",
            "relative_position_bias_table",
        ]:
            custom_keys_weight_decay.append(
                (key, args.transformer_embedding_decay)
            )
    parameters = utils.set_weight_decay(
        model,
        args.weight_decay,
        norm_weight_decay=args.norm_weight_decay,
        custom_keys_weight_decay=(
            custom_keys_weight_decay
            if len(custom_keys_weight_decay) > 0
            else None
        ),
    )

    opt_name = args.opt.lower()
    if opt_name.startswith("sgd"):
        optimizer = torch.optim.SGD(
            parameters,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov="nesterov" in opt_name,
        )
    elif opt_name == "rmsprop":
        optimizer = torch.optim.RMSprop(
            parameters,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            eps=0.0316,
            alpha=0.9,
        )
    elif opt_name == "adam":
        optimizer = torch.optim.Adam(
            parameters, lr=args.lr, weight_decay=args.weight_decay
        )
    else:
        raise RuntimeError(
            f"Invalid optimizer {args.opt}. Options SGD, RMSprop and Adam."
        )

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    args.lr_scheduler = args.lr_scheduler.lower()
    if args.lr_scheduler == "steplr":
        main_lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma
        )
    elif args.lr_scheduler == "cosineannealinglr":
        main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs - args.lr_warmup_epochs,
            eta_min=args.lr_min,
        )
    elif args.lr_scheduler == "exponentiallr":
        main_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=args.lr_gamma
        )
    else:
        raise RuntimeError(
            f"Invalid lr scheduler '{args.lr_scheduler}'. "
            "Only StepLR, CosineAnnealingLR and ExponentialLR "
            "are supported."
        )

    if args.lr_warmup_epochs > 0:
        if args.lr_warmup_method == "linear":
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=args.lr_warmup_decay,
                total_iters=args.lr_warmup_epochs,
            )
        elif args.lr_warmup_method == "constant":
            warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer,
                factor=args.lr_warmup_decay,
                total_iters=args.lr_warmup_epochs,
            )
        else:
            raise RuntimeError(
                (
                    f"Invalid warmup lr method '{args.lr_warmup_method}'. "
                    "Only linear and constant are supported."
                )
            )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_lr_scheduler, main_lr_scheduler],
            milestones=[args.lr_warmup_epochs],
        )
    else:
        lr_scheduler = main_lr_scheduler

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu]
        )
        model_without_ddp = model.module

    model_ema = None
    if args.model_ema:
        adjust = (
            args.world_size
            * args.batch_size
            * args.model_ema_steps
            / args.epochs
        )
        alpha = 1.0 - args.model_ema_decay
        alpha = min(1.0, alpha * adjust)
        model_ema = utils.ExponentialMovingAverage(
            model_without_ddp, device=device, decay=1.0 - alpha
        )

    if args.resume:
        checkpoint = torch.load(
            args.resume, map_location="cpu", weights_only=True
        )
        model_without_ddp.load_state_dict(checkpoint["model"])
        if not args.test_only:
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1
        if model_ema:
            model_ema.load_state_dict(checkpoint["model_ema"])
        if scaler:
            scaler.load_state_dict(checkpoint["scaler"])

    if args.test_only:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if model_ema:
            evaluate(
                model_ema,
                criterion,
                data_loader_test,
                device=device,
                log_suffix="EMA",
                wandb_log=args.wandb,
            )
        else:
            evaluate(
                model,
                criterion,
                data_loader_test,
                device=device,
                wandb_log=args.wandb,
            )
        return

    tune_log("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_acc1, train_acc5 = train_one_epoch(
            model,
            criterion,
            optimizer,
            data_loader,
            device,
            epoch,
            args,
            model_ema,
            scaler,
            single_training,
            wandb_log=args.wandb,
        )
        lr_scheduler.step()
        test_acc1, test_acc5 = evaluate(
            model,
            criterion,
            data_loader_test,
            device=device,
            single_training=single_training,
            wandb_log=args.wandb,
        )
        if model_ema:
            evaluate(
                model_ema,
                criterion,
                data_loader_test,
                device=device,
                log_suffix="EMA",
                wandb_log=args.wandb,
            )
        if args.output_dir:
            checkpoint = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
                "args": args,
            }
            if model_ema:
                checkpoint["model_ema"] = model_ema.state_dict()
            if scaler:
                checkpoint["scaler"] = scaler.state_dict()
            if single_training:
                utils.save_on_master(
                    checkpoint,
                    os.path.join(args.output_dir, f"model_{epoch}.pth"),
                )
                utils.save_on_master(
                    checkpoint, os.path.join(args.output_dir, "checkpoint.pth")
                )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    tune_log(f"Training time {total_time_str}")
    if single_training and args.wandb:
        wandb.finish()
    return train_acc1, train_acc5, test_acc1, test_acc5


def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(
        description="PyTorch Classification Training", add_help=add_help
    )
    # CLI Arguments
    parser.add_argument(
        "--wandb", default=False, action=argparse.BooleanOptionalAction
    )
    parser.add_argument(
        "--data-name", default="FashionMNIST", type=str, help="dataset name"
    )
    parser.add_argument(
        "--model-name", default="TwoLayerMLP", type=str, help="model name"
    )
    parser.add_argument("--lr-grid", nargs="+", type=float)
    parser.add_argument("--wd-grid", nargs="+", type=float)
    parser.add_argument(
        "--epochs",
        default=90,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )

    parser.add_argument("--opt", default="adam", type=str, help="optimizer")
    parser.add_argument(
        "--momentum", default=0.9, type=float, metavar="M", help="momentum"
    )

    parser.add_argument(
        "--norm-weight-decay",
        default=None,
        type=float,
        help="weight decay for Normalization layers (default: None)",
    )
    parser.add_argument(
        "--bias-weight-decay",
        default=None,
        type=float,
        help="weight decay for bias parameters of all layers (default: None)",
    )
    parser.add_argument(
        "--transformer-embedding-decay",
        default=None,
        type=float,
        help=(
            "weight decay for embedding parameters for vision transformer "
            "models (default: None, same value as --wd)"
        ),
    )
    parser.add_argument(
        "--label-smoothing",
        default=0.0,
        type=float,
        help="label smoothing (default: 0.0)",
        dest="label_smoothing",
    )
    parser.add_argument(
        "--mixup-alpha",
        default=0.0,
        type=float,
        help="mixup alpha (default: 0.0)",
    )
    parser.add_argument(
        "--cutmix-alpha",
        default=0.0,
        type=float,
        help="cutmix alpha (default: 0.0)",
    )
    parser.add_argument(
        "--lr-scheduler",
        default="steplr",
        type=str,
        help="the lr scheduler (default: steplr)",
    )
    parser.add_argument(
        "--lr-warmup-epochs",
        default=0,
        type=int,
        help="the number of epochs to warmup (default: 0)",
    )
    parser.add_argument(
        "--lr-warmup-method",
        default="constant",
        type=str,
        help="the warmup method (default: constant)",
    )
    parser.add_argument(
        "--lr-warmup-decay", default=0.01, type=float, help="the decay for lr"
    )
    parser.add_argument(
        "--lr-step-size",
        default=30,
        type=int,
        help="decrease lr every step-size epochs",
    )
    parser.add_argument(
        "--lr-gamma",
        default=0.1,
        type=float,
        help="decrease lr by a factor of lr-gamma",
    )
    parser.add_argument(
        "--lr-min",
        default=0.0,
        type=float,
        help="minimum lr of lr schedule (default: 0.0)",
    )
    parser.add_argument(
        "--print-freq", default=10, type=int, help="print frequency"
    )
    parser.add_argument(
        "--output-dir", default=".", type=str, help="path to save outputs"
    )
    parser.add_argument(
        "--resume", default="", type=str, help="path of checkpoint"
    )
    parser.add_argument(
        "--start-epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument(
        "--cache-dataset",
        dest="cache_dataset",
        help="Cache the datasets for quicker initialization.",
        action="store_true",
    )
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument(
        "--auto-augment",
        default=None,
        type=str,
        help="auto augment policy (default: None)",
    )
    parser.add_argument(
        "--ra-magnitude",
        default=9,
        type=int,
        help="magnitude of auto augment policy",
    )
    parser.add_argument(
        "--augmix-severity",
        default=3,
        type=int,
        help="severity of augmix policy",
    )
    parser.add_argument(
        "--random-erase",
        default=0.0,
        type=float,
        help="random erasing probability (default: 0.0)",
    )

    # Mixed precision training parameters
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use torch.cuda.amp for mixed precision training",
    )

    # distributed training parameters
    parser.add_argument(
        "--world-size",
        default=1,
        type=int,
        help="number of distributed processes",
    )
    parser.add_argument(
        "--dist-url",
        default="env://",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--model-ema",
        action="store_true",
        help="enable tracking Exponential Moving Average of model parameters",
    )
    parser.add_argument(
        "--model-ema-steps",
        type=int,
        default=32,
        help=(
            "the number of iterations that controls how often to update the "
            "EMA model (default: 32)"
        ),
    )
    parser.add_argument(
        "--model-ema-decay",
        type=float,
        default=0.99998,
        help=(
            "decay factor for Exponential Moving Average of model "
            "parameters (default: 0.99998)"
        ),
    )
    parser.add_argument(
        "--use-deterministic-algorithms",
        action="store_true",
        help="Forces the use of deterministic algorithms only.",
    )
    parser.add_argument(
        "--interpolation",
        default="bilinear",
        type=str,
        help="the interpolation method (default: bilinear)",
    )
    parser.add_argument(
        "--val-resize-size",
        default=256,
        type=int,
        help="the resize size used for validation (default: 256)",
    )
    parser.add_argument(
        "--val-crop-size",
        default=224,
        type=int,
        help="the central crop size used for validation (default: 224)",
    )
    parser.add_argument(
        "--train-crop-size",
        default=224,
        type=int,
        help="the random crop size used for training (default: 224)",
    )
    parser.add_argument(
        "--clip-grad-norm",
        default=None,
        type=float,
        help="the maximum gradient norm (default None)",
    )
    parser.add_argument(
        "--ra-sampler",
        action="store_true",
        help="whether to use Repeated Augmentation in training",
    )
    parser.add_argument(
        "--ra-reps",
        default=3,
        type=int,
        help="number of repetitions for Repeated Augmentation (default: 3)",
    )
    parser.add_argument(
        "--weights",
        default=None,
        type=str,
        help="the weights enum name to load",
    )
    parser.add_argument(
        "--backend",
        default="PIL",
        type=str.lower,
        help="PIL or tensor - case insensitive",
    )
    parser.add_argument(
        "--use-v2", action="store_true", help="Use V2 transforms"
    )

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    args.data_path = "./artifacts/data/"
    args.device = "cuda"
    args.batch_size = 64
    args.workers = 8
    main(args)

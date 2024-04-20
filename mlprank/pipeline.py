# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging as log
import os
import torch

from .architectures import registry
from .pruning import iter_modules
from .utils.evaluation import get_metrics
from .utils import get_activations, get_dataset, get_layer_list

# Pipeline Settings
LOG_LEVEL = log.INFO
LOG_HANDLE = os.path.basename(__file__)[:-3]

MLP_MODEL = registry.MLP_MODEL
DATASETS = registry.DATASETS
SCORING_MEASURES = registry.SCORING_MEASURES
PRUNING_METHODS = registry.PRUNING_METHODS

# Initialise logging
log.basicConfig()
log.root.setLevel(LOG_LEVEL)
pipe_log = log.getLogger(LOG_HANDLE)


def prune_mlp(
    model_name: str,
    dataset_name: str,
    measure_name: str = "WeightedPageRank",
    method_name: str = "Local",
    measure_args: dict = {},
    amount: float = 0.3,
    top_k: tuple = (1, 5),
    device: str = "cpu",
    batch_size: int = 64,
    out_path: str = None,
) -> tuple[dict[str, float | int], list[torch.Tensor]]:
    """
    End-to-end pruning of a pretrained neural network.

    First, it scores nodes of a pretrained multilayer perceptron using a
    registered measure and then prunes the network depending on the pruning
    method.

    Args:
    model_name (str):
        Name of the pretrained model. Must be registered in the
        `MLP_MODEL` registy.
    dataset_name (str):
        Name of the torch dataset. Must be registered in the `DATASETS`
        registy.
    measure_name (str):
        Name of the scoring measure. Must be registered in the
        `SCORING_METHODS` registy.
    method_name (str):
        Name of the pruning method. Must be registered in the
        `PRUNING_METHODS` registry.
    amount (float):
        The amount of channels to be pruned per layer or
        globally (depending on the method attribute). The value of amount
        should be a float between 0 and 1. The default is 0.3.
    top_k (tuple):
        A tuple with k values for which the top-k accuracy will be
        computed during model evaluation.
    device (str):
        The device on which operations are excecuted, e.g. 'cpu' or 'cuda'.
        The default value is 'cpu'.
    batch_size (int):
        The batch size used during evaluation and for computation of the
        calibration activations. The default value is 64.
    out_path (str):
        Specify a path to which the pruned model weights are saved.

    Returns:
    metrics (dict):
        A dictionary with metrics quantifying model performance after
        pruning is applied to the model.
    scores (list):
        List of importance score tensors. One per layer of the model.
    """

    # Verify input
    assert model_name in MLP_MODEL
    assert dataset_name in DATASETS
    assert measure_name in SCORING_MEASURES
    assert method_name in PRUNING_METHODS
    assert 0 <= amount < 1

    # Log configurations
    pipe_log.info(f"DEVICE: " + device)
    pipe_log.info(f"BATCH_SIZE: " + str(batch_size))
    pipe_log.info(f"LOG_LEVEL: " + str(LOG_LEVEL))
    pipe_log.info(f"MODEL: " + model_name)
    pipe_log.info(f"DATASET: " + dataset_name)
    pipe_log.info(f"SCORING: " + measure_name)
    pipe_log.info(f"PRUNING: " + method_name)
    pipe_log.info(f"AMOUNT: " + str(amount))

    device = torch.device(device)
    # Load base model
    pipe_log.debug("Load pretrained model")
    model = MLP_MODEL[model_name]["wrapper"]().to(device)
    # model.load_state_dict(torch.load(MLP_MODEL[model_name]["path"]))
    checkpoint = torch.load(MLP_MODEL[model_name]["path"])
    model.load_state_dict(checkpoint["model"])
    # Load data set
    pipe_log.debug("Get dataset")
    train_data, test_data = get_dataset(**DATASETS[dataset_name])
    calibration_data = torch.stack(
        [train_data[i][0] for i in range(batch_size)]
    )
    # Get parameterised modules
    layer_list = get_layer_list(model)
    if amount != 0:
        pipe_log.debug("Compute activations")
        # Compute activations on calibration data
        get_activations(model, layer_list, calibration_data, device)
        # Compute scores
        pipe_log.debug("Compute importance scores")
        layer_list = iter_modules(
            scoring_method=SCORING_MEASURES[measure_name],
            layer_list=layer_list,
            scoring_args=measure_args,
        )
        # Prune nodes
        pipe_log.debug("Prune the network")
        PRUNING_METHODS[method_name](amount, layer_list, device)
        scores = [layer["importance"] for layer in layer_list]
    else:
        scores = []

    # Evaluate model
    pipe_log.debug("Evaluate the network\n")
    metrics = get_metrics(model, layer_list, test_data, top_k, batch_size)
    pipe_log.info("METRICS:")
    for k, v in metrics.items():
        pipe_log.info(f" - {k} = {v}")
    if out_path:
        torch.save(model.state_dict(), out_path + f"/pruned_model.pth")
    return metrics, scores

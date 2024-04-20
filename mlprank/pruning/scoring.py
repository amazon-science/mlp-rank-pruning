# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging as log
import torch

from ..utils.utils import close_to_one


def iter_modules(
    scoring_method,
    layer_list: list[dict],
    scoring_args: dict = None,
) -> list[dict]:
    """
    This function sequentially iterates over all modules in `layer_list` and
    computes a score for each row in the respective weight matices.
    These scores are layer unit normalised, such that the scores of all nodes
    in one layer sum to one.
    """
    scoring_args = scoring_args or {}
    for i in range(len(layer_list)):
        weight = layer_list[i]["module"].weight
        weight = weight.abs()
        out_activations = layer_list[i]["module"].out_activation
        in_activations = layer_list[i]["module"].in_activation
        if i == 0:
            prev_importance = in_activations
        else:
            prev_importance = layer_list[i - 1]["importance"]
        prev_importance = prev_importance / prev_importance.sum()

        # Verify inputs
        assert close_to_one(prev_importance.sum())
        assert out_activations.min() >= 0

        # Compute layer score
        curr_importance = scoring_method(
            prev_importance=prev_importance,
            in_activations=in_activations,
            out_activations=out_activations,
            weight=weight,
            **scoring_args
        )
        curr_importance = curr_importance / curr_importance.sum()

        # Verify outputs
        assert close_to_one(curr_importance.sum())
        assert curr_importance.min() >= 0

        # Unit normalisation of importance score
        layer_list[i]["importance"] = curr_importance
    return layer_list


def weighted_page_rank(
    prev_importance: torch.Tensor,
    in_activations: torch.Tensor,
    out_activations: torch.Tensor,
    weight: torch.nn.Parameter,
    gamma: float = 0.9,
    theta: float = 0.5,
) -> torch.Tensor:
    """
    Computes the page-rank score for a module of the mlp network, given the
    weight matrix of the module and the importance score of the previous
    module.
    """
    assert 0 <= gamma <= 1, "The dampening factor must be between 0 and 1"
    assert 0 <= theta <= 1, "The trade-off parameter must be between 0 and 1"

    # Compute weight-degree trade off
    adjacency = torch.where(weight != 0, torch.tensor(1), torch.tensor(0))
    scaled_adjacency = adjacency / adjacency.sum(dim=0)
    scaled_weight = weight / weight.sum(dim=0)
    # Replace nan with zero
    scaled_weight = torch.nan_to_num(scaled_weight, nan=0.0)
    scaled_adjacency = torch.nan_to_num(scaled_adjacency, nan=0.0)
    # Compute convex theta combination
    weighted_matrix = (theta * scaled_weight) + (
        (1 - theta) * scaled_adjacency
    )

    # Get importance from previous layer
    curr_importance = weighted_matrix.matmul(prev_importance)

    # Apply damping factor
    curr_importance = gamma * curr_importance
    damping = ((1 - gamma) * out_activations) / out_activations.sum()
    curr_importance = curr_importance + damping
    return curr_importance


def modified_page_rank(
    prev_importance: torch.Tensor,
    in_activations: torch.Tensor,
    out_activations: torch.Tensor,
    weight: torch.nn.Parameter,
    z_importance: bool = False,
    amount: float = 0.3,
) -> torch.Tensor:
    """
    Computes the modified page-rank score for a module of the mlp network,
    given the weight matrix of the module and the importance score of the
    previous module.
    """
    assert prev_importance.sum() - 1 < 0.0001, prev_importance.sum()
    assert out_activations.min() >= 0

    if z_importance:
        # Set previously pruned nodes to zero
        n_pruned = int(in_activations.shape[0] * amount)
        prev_importance[prev_importance.argsort()[:n_pruned]] = 0

    # Get importance from previous layer
    curr_importance = weight.matmul(prev_importance)
    curr_importance = curr_importance * out_activations
    return curr_importance


def channel_random(
    prev_importance: torch.Tensor,
    in_activations: torch.Tensor,
    out_activations: torch.Tensor,
    weight: torch.nn.Parameter,
) -> torch.Tensor:
    """
    Computes the random importance scores for a module of the mlp network.
    """
    channels, inputs = weight.shape
    curr_importance = torch.rand(channels)
    return curr_importance


def channel_norm(
    prev_importance: torch.Tensor,
    in_activations: torch.Tensor,
    out_activations: torch.Tensor,
    weight: torch.nn.Parameter,
    p: int = 1,
) -> torch.Tensor:
    """
    Computes the channel p norm importance scores for a module of the mlp
    network, given the weight matrix of the module.
    previous module.
    """
    curr_importance = torch.norm(weight, dim=1, p=1)
    return curr_importance


def channel_activation(
    prev_importance: torch.Tensor,
    in_activations: torch.Tensor,
    out_activations: torch.Tensor,
    weight: torch.nn.Parameter,
    p: int = 1,
) -> torch.Tensor:
    """
    Computes the random importance scores for a module of the mlp network,
    given the out_activation of the module.
    """
    curr_importance = out_activations
    return curr_importance


def channel_wanda(
    prev_importance: torch.Tensor,
    in_activations: torch.Tensor,
    out_activations: torch.Tensor,
    weight: torch.nn.Parameter,
) -> torch.Tensor:
    """
    Computes the agregated channel wanda scores for a module of the mlp
    network, given the out_activation of the module.
    """
    curr_importance = weight.abs().matmul(in_activations)
    return curr_importance

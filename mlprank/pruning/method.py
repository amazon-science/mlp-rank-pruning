# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import math
import torch
import torch.nn.utils.prune as prune


def local_pruning(amount: float, layer_list: list[dict], device) -> None:
    """
    Does local structured pruning. In particualr it prunes the lowest scoring
    `amount` of nodes of each modules in layer list, which corresponds to
    removing rows from the weight matrices.
    """
    for layer in layer_list:
        # Reshape importance score to match weight matrix
        importance_scores = (
            layer["importance"]
            .unsqueeze(1)
            .repeat(1, layer["module"].weight.shape[1])
        )
        # Create pruning buffer
        prune.ln_structured(
            module=layer["module"],
            name="weight",
            importance_scores=importance_scores,
            amount=amount,
            dim=0,
            n=1,
        )
        # Make pruning permanent
        prune.remove(layer["module"], "weight")
    return None


def global_pruning(amount: float, layer_list: list[dict], device):
    """
    Does global structured pruning. In particualr it prunes the lowest scoring
    `amount` of nodes in the entire model. Removing one row corresponds to
    removing one row from a weight matrix.
    """
    flat_scores = torch.cat(
        [
            layer["importance"]
            for layer in layer_list
            if hasattr(layer["module"], "weight")
        ],
        dim=0,
    )
    threshold_idx = math.ceil(len(flat_scores) * amount)
    lowest_nodes = flat_scores.argsort()[:threshold_idx]
    flat_mask = torch.ones(len(flat_scores))
    for i in lowest_nodes:
        flat_mask[i] = 0
    # Reshape to pruning masks
    pointer = 0
    for layer in layer_list:
        if not hasattr(layer["module"], "weight"):
            continue
        layer_mask = flat_mask[
            pointer : pointer + layer["module"].out_features
        ]
        pointer += layer["module"].out_features
        layer["pruning_mask"] = layer_mask.unsqueeze(1).repeat(
            1, layer["module"].in_features
        )
        layer["amount"] = int(
            layer_mask.numel() - torch.count_nonzero(layer_mask)
        )
    # Use Structured Pruning
    for layer in layer_list:
        # Create pruning buffer
        prune.ln_structured(
            module=layer["module"],
            name="weight",
            importance_scores=layer["pruning_mask"].to(device),
            amount=layer["amount"],
            dim=0,
            n=1,
        )
        # Make pruning permanent
        prune.remove(layer["module"], "weight")
    return None

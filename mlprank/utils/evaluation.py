# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch
from torch.utils.data import DataLoader


def model_size(model: torch.nn.Module):
    """
    Counts the number of parameters in the model. Specifically it counts the
    total number of parameters as well as the number of non-zero parameters.
    """
    total_params = 0
    nonzero_params = 0
    for param in model.parameters():
        total_params += param.numel()
        nonzero_params += torch.count_nonzero(param)
    return int(total_params), int(nonzero_params)


def flops(layer_list):
    """
    Computes the number of floating point operations of the modules in the
    input list.
    """
    # TODO does not include ReLU operations
    total_flops = nonzero_flops = 0
    for layer in layer_list:
        module = layer["module"]
        w = module.weight.detach().cpu().numpy().copy()
        linear_flops = module.in_features * module.out_features
        total_flops += linear_flops
        zero_flops_share = np.sum(w != 0.0).sum() / np.prod(w.shape)
        nonzero_flops += linear_flops * zero_flops_share
    return total_flops, nonzero_flops


def correct(output, target, topk=(1,)):
    """
    Calculate the number of correct predictions given the model outpus as well
    the correct targetsn for all k. A prediction is correct if the target is
    included in the top k output predictions.
    """
    with torch.no_grad():
        maxk = max(topk)
        _, pred = output.topk(k=maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        topk_accuracy = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            topk_accuracy.append(correct_k.item())
        return topk_accuracy


def accuracy(model, dataloader, topk=(1,)):
    """
    Computes the top-k accuracy of the model on a given dataset for a tuple of
    different k values.
    """
    # Use same device as model
    device = next(model.parameters()).device

    accs = np.zeros(len(topk))
    with torch.no_grad():

        for i, (input, target) in enumerate(dataloader):
            input = input.to(device)
            target = target.to(device)
            output = model(input)

            accs += np.array(correct(output, target, topk))

    # Normalize over data length
    accs /= len(dataloader.dataset)
    accs_dict = {f"top{k}_acc": accs[i] for i, k in enumerate(topk)}
    return accs_dict


def get_metrics(model, layer_list, test_data, top_k, batch_size):
    """
    Computes top-k accuracy, the number of FLOPs and theoretical speedup of
    the model on the test data.
    """
    metrics = {}
    # Model Size
    size, size_nz = model_size(model)
    metrics["size"] = size
    metrics["size_nz"] = size_nz
    metrics["compression_ratio"] = size / size_nz
    metrics["actual_amount"] = 1 - (size_nz / size)
    # FLOPS
    ops, ops_nz = flops(layer_list)
    metrics["flops"] = ops
    metrics["flops_nz"] = ops_nz
    metrics["theoretical_speedup"] = ops / ops_nz
    # Accuracy
    dataloader = DataLoader(test_data, batch_size=batch_size)
    metrics = {**metrics, **accuracy(model, dataloader, top_k)}
    return metrics

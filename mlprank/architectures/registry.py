# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from torchvision import datasets

from . import models
from ..pruning import scoring
from ..pruning import method

# Registry
base_path = "./artifacts/"

MLP_MODEL = {
    "TwoLayerMLP": {
        "wrapper": models.TwoLayerMLP,
        "path": base_path + "models/TwoLayerMLP/model_19.pth",
    },
    "ThreeLayerMLP": {
        "wrapper": models.ThreeLayerMLP,
        "path": base_path + "models/ThreeLayerMLP/model_99.pth",
    },
    "SixLayerMLP": {
        "wrapper": models.SixLayerMLP,
        "path": base_path + "models/SixLayerMLP/model_199.pth",
    },
    "TwelveLayerMLP": {
        "wrapper": models.TwelveLayerMLP,
        "path": base_path + "models/TwelveLayerMLP/model_199.pth",
    },
}
DATASETS = {
    "MNIST": {"wrapper": datasets.MNIST, "path": base_path + "data/"},
    "FashionMNIST": {
        "wrapper": datasets.FashionMNIST,
        "path": base_path + "data/",
    },
    "CIFAR10": {"wrapper": datasets.CIFAR10, "path": base_path + "data/"},
    "CIFAR100": {"wrapper": datasets.CIFAR100, "path": base_path + "data/"},
}
SCORING_MEASURES = {
    "WeightedPageRank": scoring.weighted_page_rank,
    "ModifiedPageRank": scoring.modified_page_rank,
    "ChannelRandom": scoring.channel_random,
    "ChannelNorm": scoring.channel_norm,
    "ChannelActivation": scoring.channel_activation,
    "ChannelWandA": scoring.channel_wanda,
}
PRUNING_METHODS = {
    "Local": method.local_pruning,
    "Global": method.global_pruning,
}
# On which dataset the model was trained
MODEL_DATA = {
    "TwoLayerMLP": "MNIST",
    "ThreeLayerMLP": "FashionMNIST",
    "SixLayerMLP": "CIFAR10",
    "TwelveLayerMLP": "CIFAR100",
}

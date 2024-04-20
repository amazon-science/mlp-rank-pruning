# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import torchvision.transforms as transforms


def get_dataset(wrapper, path, augment=False) -> tuple:
    """
    Loads a torchvision dataset and returns training and testing data.
    """
    # Define processing
    if augment:
        transform = transforms.Compose(
            [
                transforms.AutoAugment(
                    transforms.AutoAugmentPolicy.CIFAR10,
                    interpolation=transforms.InterpolationMode.NEAREST,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    else:
        transform = transforms.ToTensor()
    # Load data
    training_data = wrapper(
        root=path, train=True, download=True, transform=transform
    )
    test_data = wrapper(
        root=path, train=False, download=True, transform=transform
    )
    return training_data, test_data


def get_activations(model, layer_list, calibration_data, device) -> None:
    """
    Computes calibration activations of a model given a data sample.
    """
    relu = torch.nn.ReLU()

    # Define forward hook
    def hook_fn(module, input, output):
        # module.out_activation = relu(output).mean(axis=0)
        # module.in_activation = input[0].mean(axis=0)
        module.out_activation = torch.norm(relu(output), p=2, dim=0)
        module.in_activation = torch.norm(input[0], p=2, dim=0)

    hooks = []
    for layer in layer_list:
        hooks.append(layer["module"].register_forward_hook(hook_fn))
    model(calibration_data.to(device))
    for hook in hooks:
        hook.remove()
    return None


def get_layer_list(model) -> list[dict]:
    """
    Takes a pytorch model as input and returns a list of dictionaries with the
    linear layers and their name.
    """
    layer_list = []
    # Iterate modules of the model
    for name, module in dict(model.named_modules()).items():
        if module.__class__ == torch.nn.Linear:
            layer_list.append({"name": name, "module": module})
    return layer_list


def close_to_one(n, tol: float = 0.0001) -> bool:
    """
    Checks if a number is close to one with respect to the tol parameter.
    """
    return abs(n - 1) < tol

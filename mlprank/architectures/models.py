# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn


class TwoLayerMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 256), nn.ReLU(), nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class ThreeLayerMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class SixLayerMLP(nn.Module):
    def __init__(self, input_size=3 * 32 * 32, num_class=10):
        super(SixLayerMLP, self).__init__()
        self.fc1 = nn.Linear(in_features=input_size, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=512)
        self.fc3 = nn.Linear(in_features=512, out_features=256)
        self.fc4 = nn.Linear(in_features=256, out_features=256)
        self.fc5 = nn.Linear(in_features=256, out_features=128)
        self.fc_last = nn.Linear(in_features=128, out_features=num_class)
        self.batchnorm = nn.BatchNorm1d(512)
        self.batchnorm2 = nn.BatchNorm1d(256)
        self.batchnorm3 = nn.BatchNorm1d(128)

    def forward(self, x):
        x = torch.flatten(x, 1)
        first = self.batchnorm(nn.functional.relu(self.fc1(x)))
        second = self.batchnorm(nn.functional.relu(self.fc2(first))) + first
        third = self.batchnorm2(nn.functional.relu(self.fc3(second)))
        fourth = self.batchnorm2(nn.functional.relu(self.fc4(third))) + third
        fith = self.batchnorm3(nn.functional.relu(self.fc5(fourth)))
        out = self.fc_last(fith)
        return out


class TwelveLayerMLP(nn.Module):
    def __init__(self, input_size=3 * 32 * 32, num_class=100):
        super(TwelveLayerMLP, self).__init__()
        self.fc1 = nn.Linear(in_features=input_size, out_features=2048)
        self.fc2 = nn.Linear(in_features=2048, out_features=2048)
        self.fc3 = nn.Linear(in_features=2048, out_features=1024)
        self.fc4 = nn.Linear(in_features=1024, out_features=1024)
        self.fc5 = nn.Linear(in_features=1024, out_features=1024)
        self.fc6 = nn.Linear(in_features=1024, out_features=1024)
        self.fc7 = nn.Linear(in_features=1024, out_features=512)
        self.fc8 = nn.Linear(in_features=512, out_features=512)
        self.fc9 = nn.Linear(in_features=512, out_features=256)
        self.fc10 = nn.Linear(in_features=256, out_features=256)
        self.fc11 = nn.Linear(in_features=256, out_features=256)
        self.fc_last = nn.Linear(in_features=256, out_features=num_class)
        self.batchnorm2048 = nn.BatchNorm1d(2048)
        self.batchnorm1024 = nn.BatchNorm1d(1024)
        self.batchnorm512 = nn.BatchNorm1d(512)
        self.batchnorm256 = nn.BatchNorm1d(256)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.batchnorm2048(nn.functional.relu(self.fc1(x)))
        x = self.batchnorm2048(nn.functional.relu(self.fc2(x)))
        x = self.batchnorm1024(nn.functional.relu(self.fc3(x)))
        x = self.batchnorm1024(nn.functional.relu(self.fc4(x)))
        x = self.batchnorm1024(nn.functional.relu(self.fc5(x)))
        x = self.batchnorm1024(nn.functional.relu(self.fc6(x)))
        x = self.batchnorm512(nn.functional.relu(self.fc7(x)))
        x = self.batchnorm512(nn.functional.relu(self.fc8(x)))
        x = self.batchnorm256(nn.functional.relu(self.fc9(x)))
        x = self.batchnorm256(nn.functional.relu(self.fc10(x)))
        x = self.batchnorm256(nn.functional.relu(self.fc11(x)))
        x = self.fc_last(x)
        return x

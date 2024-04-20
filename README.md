# MLP-Rank: A Graph Theoretical Approach to Pruning Deep Neural Networks

This is the PyTorch implementation of pruning neural networks based on the weighted PageRank centrality measure as introduced by my thesis:

**MLP-Rank: A Graph Theoretical Approach to Pruning Deep Neural Networks** \
Amazon Web Services - AI Research and Education \
Author: David B. Hoffmann \
Advisor: Dr. Kailash Budhathoki, Dr. Matthäus Kleindessner \
[Arxiv Paper](add/link/to/paper.de) 

```bibtex
@mastersthesis{hoffmann2024mlprank,
    type={Bachelor's Thesis},
    title={MLP-Rank: A Graph Theoretical Approach to Pruning Deep Neural Networks},
    author={Hoffmann, David B.},
    year={2024},
}
```

## Setup

To setup mlp-rank-pruning, clone the repository and install it locally with `pip install -e .`

## How To Use

The package provides everything needed to train and prune mulilayer perceptron networks as done in the thesis. It contains a **training** module which can be used to conduct distributed hyperparameter optimisation over a simple grid as well as training of single MLP model, for image classification tasks. For further documentation related to training refer to the [training README](https://gitlab.aws.dev/adavidho/mlp-rank-pruning/-/blob/main/training/README.md?ref_type=heads). Note that the code in the training folder is subject to a separate license, which is provided in that folder. \
The **mlprank** package contains everything related to pruning a pretrained neural network. It is designed to be easily extendible and compatible with further methods and scoring functions for structured pruning to allow for other comparisons, which is further detailed in the [architecture README](https://gitlab.aws.dev/adavidho/mlp-rank-pruning/-/blob/main/mlprank/architectures/README.md?ref_type=heads). The core function of the mlprank package is prune_mlp, which is documented in the [mlprank README](https://gitlab.aws.dev/adavidho/mlp-rank-pruning/-/blob/main/mlprank/README.md?ref_type=heads). To do structured pruning using weighted PageRank centrality on a simple three layer MLP using MNIST as calibration data, the following code can be used: 
```python
from mlprank import prune_mlp

metrics, scores = prune_mlp(
    model_name="TwoLayerMLP",
    dataset_name="MNIST",
    method_name="Local",
    measure_name="WeightedPageRank",
    amount=0.3,
    out_path="."
)
```
The **experiment** module acts as a CLI tool for running multiple experiments. It takes a list of fixed and iterable arguments for different networks, scoring functions, pruning methods and pruning amounts and calls the `mlprank.prune_mlp` method. For more details refer to the [experiment README](https://gitlab.aws.dev/adavidho/mlp-rank-pruning/-/blob/main/experiments/README.md?ref_type=heads). 

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.


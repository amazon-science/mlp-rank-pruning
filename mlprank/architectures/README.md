# Support for Model Architectures

## Description 

This module contains everything that is related to supporting a particular model. 
All supported neural networks are defined in `models.py`. They are, however, not directly accessed by the pipeline and other modules. 
Instead, they have to be registered in `registry.py` together with other information such as the path to `checkpoint.pth` file and the dataset used for calibration data and final evaluation. 
All related datasets must be available through the torchvision datasets API and additionally have to be registered in `registry.py`.
The same is true for scoring methods and pruning functions. 

As this is the case, the `registry.py` file provides a comprehensive overview of all current functionality and options of the mlp-rank-pruning package in a single location. 

## Adding a Model

To add a new model based on a simple MLP architecture, follow the steps below:

1. Define the model structure in `models.py` with a unique and descriptive class name.
2. Add an entry to the `MLP_MODEL` dictionary in `registry.py` with the following structure: `"<ModelName>:{"wrapper":<models.ModelClass>}`.
3. Add the dataset the model was trained on to the `DATASETS` dictionary in  `registry.py` with the following structure: `"<DataName>:{"wrapper":<datasets.DataClass>, "path": <path/to/local/dataset>}`.
4. Train the model as described in the [Tuning and Training MLP Models](https://gitlab.aws.dev/adavidho/mlp-rank-pruning/-/blob/main/training/README.md?ref_type=heads) documentation.
5. Add the path to trained model weights as an attribute to the `MLP_MODEL` dictionary in `registry.py` with the following structure: `"path": <path/to/model/checkpoint.pth>`.
6. Lastly, add an entry to `MODEL_DATA` mapping the model name to the name of the dataset it was trained on as they were referenced in `MLP_MODEL` and `DATASETS` respectively.

## Adding a Scoring Function

To add a new function for computing importance scores, follow the steps below:

1. Define the scoring function in `../pruning/scoring.py` with a unique and descriptive function name. It must accept prev_importance, in_activations, out_activations and weight as arguments, but can also receive additional parameters through kwargs. The function must return a tensor of shape (n_neurons,) for each layer.
2. Add an entry to the `SCORING_MEASURES` dictionary in the `registry.py` file with the following structure: `<scoring_name><scoring.your_scoring_function>`.

## Adding a Pruning Method

To define a new pruning method which uses the importance scores computed by the scoring function, adhere to the following steps:

1. Define the scoring function in `../pruning/method.py` with a unique and descriptive function name. It must accept amount, layer_list and device as arguments.
2. Add an entry to the `PRUNING_METHODS` dictionary in the `registry.py` file with the following structure: `<method_name><method.your_method_function>`.

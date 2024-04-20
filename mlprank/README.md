## Arguments

**model_name (str):**  \
Name of the pretrained model. Must be registered in the `MLP_MODEL` registry. \
*Options:* "TwoLayerMLP", "ThreeLayerMLP", "SixLayerMLP", "TwelveLayerMLP".
  
**dataset_name (str):**  \
Name of the torch dataset. Must be registered in the `DATASETS` registry. \
*Options:* "MNIST", "FashionMNIST", "CIFAR10", "CIFAR100".
  
**measure_name (str):**  \
Name of the scoring measure. Must be registered in the `SCORING_METHODS` registry. \
*Options:* "WeightedPageRank", "ModifiedPageRank", "ChannelRandom", "ChannelNorm", "ChannelActivation", "ChannelWandA".  
*Default:* "WeightedPageRank"
  
**method_name (str):**  \
Name of the pruning method. Must be registered in the `PRUNING_METHODS` registry. \
*Options:* "Local", "Global".  \
*Default:* "Local"
  
**measure_args (dict):**  \
Additional arguments to be passed to the scoring method.  \
*Default:* {} \
*Example:* When running WeightedPageRank with cusom gamma and theta and ChannelNorm with cusom p we can use: \
`{"ChannelNorm": {"p": 1}, "WeightedPageRank": {"gamma": 0.5, "theta": 1}}`

**amount (float):**  \
The amount of channels to be pruned per layer or globally (depending on the method attribute). \
The value of amount should be a float between 0 and 1.  \
*Default:* 0.3

**top_k (tuple):**  \
A tuple with k values for which the top-k accuracy will be computed during model evaluation. \
*Default:* (1, 5)

**device (str):**  \
The device on which operations are executed, e.g., 'cpu' or 'cuda'. \
*Default:* "cpu"

**batch_size (int):**  \
The batch size used during evaluation and for computation of the calibration activations.  \
*Default:* 64

**out_path (str):**  \
Specify a path to which the pruned model weights are saved.
# Pruning Experiments

The `experiments.py` script calls the `mlprank.prune_mlp` function for a series of different networks, scoring functions, pruning methods and amounts of pruning. It is a CLI tool that accepts the following arguments:

**--model-name:** \
*Description:* Specifies the name(s) of the model(s) to be pruned. \
*Usage:* Provide one or more model registry names as input.

**--method-name:** \
*Description:* Specifies the type(s) of pruning method(s) to be applied (e.g., Local or Global). \
*Usage:* Provide one or more pruning type names as input.

**--measure-name:** \
*Description:* Specifies the name(s) of the measure(s) used for pruning. \
*Usage:* Provide one or more measure registry names as input.

**--amount:** \
*Description:* Specifies the share of nodes to be pruned as a percentage. \
*Usage:* Provide one or more floating-point values representing the share of nodes to be pruned.

**--measure-args:** \
*Description:* Specifies additional arguments to be passed to the scoring method. \
*Usage:* Provide additional arguments in JSON format. These arguments will be passed to the scoring method. \

**--wandb:** \
*Description:* Enables logging via wandb (Weights & Biases). \
*Usage:* Include this flag to enable logging via wandb.

**--out-path:** \
*Description:* Specifies the relative output path for storing metrics and scores. \
*Usage:* Provide the desired relative output path as a string. If not specified, it defaults to "./experiments/results".



# Examples

The following command is used to run all pruning variations found in the paper:
```sh
python experiments/experiments.py \
    --model-name TwoLayerMLP ThreeLayerMLP SixLayerMLP TwelveLayerMLP\
    --method-name Local Global \
    --measure-name \
        WeightedPageRank \
        ModifiedPageRank \
        ChannelRandom \
        ChannelNorm \
        ChannelActivation \
        ChannelWandA \
    --amount 0.1 0.2 0.3 0.4 0.5 \
    --wandb
```

To evaluate the dense baseline for each model, this comand is used:
```sh
python experiments/experiments.py \
    --model-name TwoLayerMLP ThreeLayerMLP SixLayerMLP TwelveLayerMLP\
    --method-name Local \
    --measure-name ChannelRandom \
    --amount 0 \
    --wandb
```


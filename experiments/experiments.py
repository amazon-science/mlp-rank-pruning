import argparse
import json
import logging as log
import os
import wandb
import torch

from mlprank import registry
from mlprank import prune_mlp


# Experiment Settings
MODEL_DATA = registry.MODEL_DATA
LOG_LEVEL = log.INFO
LOG_HANDLE = os.path.basename(__file__)[:-3]

# Initialise logging
log.basicConfig()
log.root.setLevel(LOG_LEVEL)
exp_log = log.getLogger(LOG_HANDLE)


def main(args):
    """
    Iterates over all configuration variables of a series of experiments,
    automatically calls `mlprank.prune_mlp' with the particualr configuration
    and logs the returned results.
    This function is used as a CLI tool in combination with argparse.
    """
    for model in args.model_name:
        for measure in args.measure_name:
            for method in args.method_name:
                for amount in args.amount:
                    data = MODEL_DATA[model]
                    try:
                        # Start pruning process
                        if measure in args.measure_args:
                            measure_args = args.measure_args[measure]
                        elif measure == "ModifiedPageRankZimp":
                            measure_args = {"amount": amount}
                        else:
                            measure_args = None
                        metrics, scores = prune_mlp(
                            model_name=model,
                            dataset_name=data,
                            method_name=method,
                            measure_name=measure,
                            measure_args=measure_args,
                            amount=amount,
                        )
                        # Log results
                        group_name = f"Pruning-{model}-{data}"
                        experiment_name = group_name + (
                            f"-{measure}"
                            f"-{method}"
                            f"-{str(amount).replace('.', '_')}"
                        )
                        if args.wandb:
                            wandb.init(
                                project="MLP-Pruning",
                                group=group_name,
                                name=experiment_name,
                                config={
                                    "measure": measure,
                                    "method": method,
                                    "amount": amount,
                                    "scores": scores,
                                },
                            )
                            # Log experiment results
                            wandb.log(metrics)
                            wandb.finish()
                        else:
                            exp_log.info(f"{experiment_name}: {metrics}")
                        # Save experiment results to json
                        base_path = (
                            f"{args.out_path}/{group_name}/{experiment_name}"
                        )
                        if not os.path.exists(base_path):
                            os.makedirs(base_path)
                        with open(
                            f"{base_path}/metric.json", "w"
                        ) as m_outfile:
                            json.dump(metrics, m_outfile)
                        torch.save(scores, f"{base_path}/scores.json")
                    except Exception as e:
                        exp_log.error(f"ERROR: Experiment Failed! {e}")


def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(
        description="MLP-Rank-Pruning Experiment Series", add_help=add_help
    )
    # Script Arguments
    parser.add_argument("--model-name", nargs="+", help="Model registry name")
    parser.add_argument(
        "--method-name",
        nargs="+",
        help="Name of the pruning type (Local, Global)",
    )
    parser.add_argument(
        "--measure-name", nargs="+", help="Measure registry name"
    )
    parser.add_argument(
        "--amount", nargs="+", type=float, help="Share of nodes pruned"
    )
    parser.add_argument(
        "--measure-args",
        default={},
        type=json.loads,
        help="Args passed to the scoring method",
    )
    parser.add_argument(
        "--wandb",
        default=False,
        action="store_true",
        help="Enables logging via wandb",
    )
    parser.add_argument(
        "--out-path",
        default="./experiments/results",
        type=str,
        help="Relative output path for metrics and scores",
    )
    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)

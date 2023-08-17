# Reproducing TableShift Benchmark Results

This document describes how to reproduce the results from our main paper, *Benchmarking Distribution Shift in Tabular Data with TableShift*. If you instead want to evaluate your own algorithm(s) on the TableShift benchmark, see the `benchmarking_guide.md` in this repo.

**Note:** Running the experiments for this paper consumed a significant amount of computational resources (thousands of GPU-hours on a large cluster, plus thousands of CPU-hours for CPU-only models). In order to reproduce our results in full, you will need access to similar computational resources.

## Environment Setup

First, ensure your environment is set up properly.

Follow the installation instructions on the main README to set up your environment.

As a check of your setup, make sure that you're able to run the `examples/run_expt.py` script.

## Dataset Access

There are 15 datasets in the TableShift benchmark. All of the datasets are publicly accessible (although a few require public credentialized access). Instructions for accessing these datasets are in the `docs` directory of this repo or on the TableShift website [datasets](https://tableshift.org/datasets.html) page. 

To fully reproduce our results, you will need access to all 15 of the bennchmark datasets. However, to begin with, we recommend starting with a public dataset (those marked as "Public" on the [datasets](https://tableshift.org/datasets.html) page). If you decide to use a public credentialized dataset instead, follow the instructions on the [datasets](https://tableshift.org/datasets.html) page to access the data, download any data file(s), and place them in the TableShift cache directory (by default, this will be `tableshift/tmp`).

## Running The Experiments

There are 15 benchmarks and 19 learning algorithms. However, the procedure for these is identical.

Ensure you are in an environment with the TableShift dependencies installed. Then, run the following:
```
python scripts/ray_train.py \
    --models mlp \
    --experiment diabetes_readmission
```

You can substitute other models and dataset names for `mlp` and `diabetes_readmission` respectively.

The full list of benchmark dataset names (e.g. `diabetes_readmission` in the example above) is given in the main README. For more details, see the [datasets](https://tableshift.org/datasets.html) page or our paper.

The full list of model names is given below. For more details on each algorithm, see our paper.


| Model                 | Name in Tableshift |
|-----------------------|--------------------|
 | CatBoost              | `catboost`         |
| XGBoost               | `xgb`              |
| LightGBM              | `lightgbm`         |
| SAINT                 | `saint`            |
| NODE                  | `node`             |
| Group DRO             | `group_dro`        |
| MLP                   | `mlp`              |
| Tabular ResNet        | `resnet`           |
| Adversarial Label DRO | `aldro`            |
| CORAL                 | `deepcoral`        |
| MMD                   | `mmd`              | 
| DRO                   | `dro`              |
| DANN                  | `dann`             | 
| TabTransformer        | `tabtransformer`   |
| MixUp                 | `mixup`            |
| Label Group DRO       | `label_group_dro`  |
| IRM                   | `irm`              |
| VREX                  | `vrex`             |
| FT-Transformer        | `ft_transformer`   | # ??
 
| test                 | test |

test
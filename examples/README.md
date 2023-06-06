# TableShift Examples

This directory contains example scripts to demonstrate the use of TableShift.

**Quickstart**: to run a single experiment to test your setup, from within the `tableshift` conda environment simply run

``` 
python run_expt.py --experiment adult --model histgbm
```

## Caching Data

*(optional, but suggested!)*

**Relevant script: `scripts/cache_task.py`**

Before running any experiment, it is recommended to cache the data. Caching fetches any remote data sources, applies preprocessing and splitting, and writes the dataset to disk in a set of sharded files. This allows for more efficient training in distributed settings and for using datasets too large to fit in-memory for example when training large models on a GPU.

To cache a dataset, just use the provided script:

```
python scripts/cache_task.py --experiment $EXPERIMENT
```

This will write a set of files to the TableShift cache directory (by defailt, `tableshift/tmp`).

## Training a Single Model

**Relevant script: `run_expt.py`**

To train a single model with a fixed set of hyperparameters (for example, for testing/debugging your setup, or trying a newly-implemented model), we provide a simple script to run training. This script supports both PyTorch-based and scikit-learn-based training. 

For example, the following command will execute training on the UCI Adult dataset with an XGBoost model: 

``` 
python run_expt.py --experiment adult --model xgb
```

and this will train on the UCI Adult dataset with tabular ResNet:

``` 
python run_expt.py --experiment adult --model resnet
```

## Running Distributed Hyperparameter Tuning Experiments

In most cases, you will eventually want to tune a model's hyperparameters to ensure that it achieves good performance. We conduct hyperparameter tuning with Ray.

To conduct a toy hyperparameter tuning run, use the provided `ray_train_example.sh` or directly call the `ray_train.py` script via:

``` 
ulimit -u 127590 && python scripts/ray_train.py \
	--experiment adult \
	--num_samples 2 \
	--num_workers 1 \
	--cpu_per_worker 4 \
	--use_cached \
	--models xgb
```

The optimal configuration for Ray depends on your available system resources, and the size of the dataset and model being trained. We cannot provide guidance on configuring your experiments and would direct users with questions about Ray to the [Ray Tune docs](https://docs.ray.io/en/latest/tune/index.html).

## Available Models and Identifiers

To train a model, you need to use the correct identifier as an argument to the `--model` flag. A complete list of supported models and their identifiers is below.

| Model                                      | Identifier        |
|--------------------------------------------|-------------------|
| Adversarial Label DRO                      | `aldro`           |
| DeepCORAL                                  | `deepcoral`       |
| Distributionally Robust Optimization (DRO) | `dro`             |
| Domain MixUp                               | `mixup`           |
| Domain Adversarial Neural Networks         | `dann`            |
| FT-Transformer                             | `ft_transformer`  |
| Group DRO (domains as 'groups')            | `group_dro`       |
| Group DRO (labels/classes as 'groups')     | `label_group_dro` |
| HistGBM                                    | `histgbm`         |
| Invariant Risk Minimization                | `irm`             |
| LightGBM                                   | `lightgbm`        |
| Maximum Mean Discrepancy (MMD)             | `mmd`             |
| MLP                                        | `mlp`             |
| Neural Oblivious Decision Ensembles (NODE) | `node`            |
| Risk Extrapolation (V-REx)                 | `vrex`            |
| SAINT                                      | `saint`           |
| TabTransformer                             | `tabtransformer`  |
| Tabular ResNet                             | `resnet`          |
| XGBoost                                    | `xgb`             |

For details on each model, see the paper; implementations for each model are in the `tableshift.models` module.

For CatBoost, see below.

## Using CatBoost

As of the release of this paper, CatBoost is not officially supported in Ray. In order to train and tune CatBoost models, we provide a separate script to train CatBoost using Optuna.

To train and tune a CatBoost model:

``` 
python scripts/train_catboost_optuna.py --experiment $EXPERIMENT --use_cached --num_samples 100
```

See the script for more information about available flags, including GPU support.

# Implementing a New Model

If you would like to add your own PyTorch model to the TableShift benchmark, for example to experiment with improved robustness to distribution shift, you will need to do the following:

* Implement a model class that subclasses `tableshift.models.compat.SklearnStylePytorchModel` and implements all relevant methods.
* Add the model's string identifier to the `PYTORCH_MODEL_NAMES` constant in `tableshift.models.compat`
* Modify `tableshift.models.utils.get_estimator()` to return a class of the new model
* (If hyperparameter tuning is desired): Modify `tableshift.configs.hparams.py` to include a search space for your model.

If you are interested in contributing your model to TableShift, please submit a PR to the repo.

To implement models that are *not* subclasses of `torch.nn.Module`, the model needs to provide a scikit-learn-style interface (it should implement methods `fit()`, `predict()`, and `predict_proba()`) and should instead be added to `tableshift.models.compat.SKLEARN_MODEL_NAMES`.
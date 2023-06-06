"""
Usage:

python scripts/train_catboost_optuna \
    --experiment adult \
    --use_gpu \
    --use_cached

"""
import argparse
import logging
import os

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, \
    average_precision_score

from tableshift.core import get_dataset
from tableshift.core.utils import timestamp_as_int


def evaluate(model: CatBoostClassifier, X: pd.DataFrame, y: pd.Series,
             split: str) -> dict:
    yhat_hard = model.predict(X)
    yhat_soft = model.predict_proba(X)[:, 1]
    metrics = {}
    metrics[f"{split}_accuracy"] = accuracy_score(y, yhat_hard)
    metrics[f"{split}_auc"] = roc_auc_score(y, yhat_soft)
    metrics[f"{split}_map"] = average_precision_score(y, yhat_soft)

    metrics[f"{split}_num_samples"] = len(y)
    metrics[f"{split}_ymean"] = np.mean(y).item()
    return metrics


def main(experiment: str, cache_dir: str, results_dir: str, num_samples: int,
         use_gpu: bool, use_cached: bool):
    start_time = timestamp_as_int()

    dset = get_dataset(experiment, cache_dir, use_cached=use_cached)
    uid = dset.uid

    X_tr, y_tr, _, _ = dset.get_pandas("train")
    X_val, y_val, _, _ = dset.get_pandas("validation")

    def optimize_hp(trial: optuna.trial.Trial):
        cb_params = {
            # Same tuning grid as https://arxiv.org/abs/2106.11959,
            # see supplementary section F.4.
            'learning_rate': trial.suggest_float('learning_rate', 1e-3,
                                                 1., log=True),
            'depth': trial.suggest_int('depth', 3, 10),
            'bagging_temperature': trial.suggest_float(
                'bagging_temperature', 1e-6, 1., log=True),
            'l2_leaf_reg': trial.suggest_int('l2_leaf_reg', 1, 100, log=True),
            'leaf_estimation_iterations': trial.suggest_int(
                'leaf_estimation_iterations', 1, 10),

            "use_best_model": True,
            "task_type": "GPU" if use_gpu else "CPU",
            'random_seed': 42
        }

        model = CatBoostClassifier(**cb_params)
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)
        y_pred = model.predict(X_val)
        return accuracy_score(y_val, y_pred)

    study = optuna.create_study(direction="maximize")
    study.optimize(optimize_hp, n_trials=num_samples)
    print('Trials:', len(study.trials))
    print('Best parameters:', study.best_trial.params)
    print('Best score:', study.best_value)
    print("training completed! retraining model with best params and "
          "evaluating it.")

    clf_with_best_params = CatBoostClassifier(**study.best_trial.params)
    clf_with_best_params = clf_with_best_params.fit(X_tr, y_tr)

    expt_results_dir = os.path.join(results_dir, experiment, str(start_time))

    metrics = {}
    model_name = "catboost"
    metrics["estimator"] = model_name
    metrics["domain_split_varname"] = dset.domain_split_varname
    metrics["domain_split_ood_values"] = str(dset.get_domains("ood_test"))
    metrics["domain_split_id_values"] = str(dset.get_domains("id_test"))

    splits = (
        'id_test', 'ood_test', 'ood_validation', 'validation') if dset.is_domain_split else (
        'test', 'validation')
    for split in splits:
        X, y, _, _ = dset.get_pandas(split)
        _metrics = evaluate(clf_with_best_params, X, y, split)
        print(_metrics)
        metrics.update(_metrics)

    iter_fp = os.path.join(
        expt_results_dir,
        f"tune_results_{experiment}_{start_time}_{uid[:100]}_"
        f"{model_name}.csv")
    if not os.path.exists(expt_results_dir):
        os.makedirs(expt_results_dir)

    logging.info(f"writing results for {model_name} to {iter_fp}")
    pd.DataFrame(metrics, index=[1]).to_csv(iter_fp, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", default="tmp",
                        help="Directory to cache raw data files to.")
    parser.add_argument("--experiment", default="adult",
                        help="Experiment to run. Overridden when debug=True.")
    parser.add_argument("--num_samples", type=int, default=100,
                        help="Number of hparam samples to take in tuning "
                             "sweep.")
    parser.add_argument("--results_dir", default="./optuna_results",
                        help="where to write results. CSVs will be written to "
                             "experiment-specific subdirectories within this "
                             "directory.")
    parser.add_argument("--use_cached", default=False, action="store_true",
                        help="whether to use cached data.")
    parser.add_argument("--use_gpu", action="store_true", default=False,
                        help="whether to use GPU (if available)")
    args = parser.parse_args()
    main(**vars(args))

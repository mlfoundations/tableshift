from collections.abc import Iterable
import functools
import logging

import fairlearn.metrics
import numpy as np
import pandas as pd
from torch.nn.functional import binary_cross_entropy
import torch
import sklearn.metrics as skm


def extract_positive_class_predictions(y_pred):
    if len(y_pred.shape) == 2:
        # Case: predictions contain class preds for classes (0,1).
        return y_pred[:, 1]
    else:
        # Case: predictions contain class preds only for class 1.
        return y_pred


def clip_torch_outputs(t: torch.Tensor, eps=1e-8, clip_max=1.0, clip_min=0.0):
    """Helper function to safely clip tensors with values outside a range.

    This is mostly used when casting numpy arrays to torch doubles, which can
    result in values slightly outside the expected range in critical ways
    (e.g. 1. can be cast as a double to 1.0000000000000002, which raises
    errors in binary cross-entropy).
    """
    if not (t.max() <= clip_max + eps) and (t.min() >= clip_max - eps):
        logging.warning(
            f"tensor values outside clip range [{clip_min}-{eps},{clip_max}+{eps}]")
    return torch.clip(t, min=clip_min, max=clip_max)


def all_subgroups_contain_all_label_values(y_true, sens) -> bool:
    """Check whether all labels are represented in all sensitive subgroups."""
    if np.ndim(sens) == 2:
        # Case: multiple sensitive attributes
        sens_cols = [sens.iloc[:, i] for i in range(sens.shape[1])]
        crosstab = pd.crosstab(y_true, sens_cols)
    else:
        # Case: single sensitive attribute
        crosstab = pd.crosstab(y_true, sens)
    return np.all(crosstab != 0)


def _intersectional_metrics_from_grouped_metrics(grouped_metrics, metrics,
                                                 suffix,
                                                 sensitive_features):
    # Compute metrics by subgroup (intersectional); note that marginals can
    # always be recovered using the per-group counts.
    for sens_idx, metrics_dict in grouped_metrics.by_group.to_dict(
            'index').items():
        if not isinstance(sens_idx, Iterable):
            # Case: only a single sensitive attribute.
            sens_idx = [sens_idx]
        # sens_Str is e.g. 'race0sex1'
        sens_str = ''.join(f"{col}{val}" for col, val in
                           zip(sensitive_features.columns, sens_idx))

        for metric_name, metric_value in metrics_dict.items():
            metrics[sens_str + metric_name + suffix] = metric_value
    return metrics


# DORO; adapted from
# https://github.com/RuntianZ/doro/blob/master/wilds-exp/algorithms/doro.py
def cvar_doro_criterion(outputs, targets, eps, alpha):
    batch_size = len(targets)
    loss = binary_cross_entropy(outputs, targets, reduction="none")
    # CVaR-DORO
    gamma = eps + alpha * (1 - eps)
    n1 = int(gamma * batch_size)
    n2 = int(eps * batch_size)
    rk = torch.argsort(loss, descending=True)
    loss = loss[rk[n2:n1]].sum() / alpha / (batch_size - n2)
    return loss


def cvar_doro_metric(y_true, y_pred, eps=0.005, alpha=0.2) -> float:
    """Compute CVaR DORO metric with a fairlearn-compatible interface."""
    y_pred = extract_positive_class_predictions(y_pred)
    outputs_clipped = clip_torch_outputs(torch.from_numpy(y_pred).double())
    targets_clipped = clip_torch_outputs(torch.from_numpy(y_true).double())

    return cvar_doro_criterion(outputs=outputs_clipped,
                               targets=targets_clipped,
                               eps=eps,
                               alpha=alpha).detach().cpu().numpy().item()


def cvar_metric(y_true, y_pred, alpha=0.2) -> float:
    """Compute CVaR metric with a fairlearn-compatible interface."""
    y_pred = extract_positive_class_predictions(y_pred)
    outputs_clipped = clip_torch_outputs(torch.from_numpy(y_pred).double())
    targets_clipped = clip_torch_outputs(torch.from_numpy(y_true).double())
    return cvar_doro_criterion(outputs=outputs_clipped,
                               targets=targets_clipped,
                               eps=0.,
                               alpha=alpha).detach().cpu().numpy().item()


def loss_variance_metric(y_true, y_pred):
    """Compute loss variance metric with a fairlearn-compatible interface."""
    if len(y_pred.shape) == 2:
        y_pred = y_pred[:, 1]
    elementwise_loss = binary_cross_entropy(
        input=torch.from_numpy(y_pred).float(),
        target=torch.from_numpy(y_true).float(),
        reduction="none")
    return torch.var(elementwise_loss).cpu().numpy().item()


def append_suffix_to_keys(d: dict, suffix: str) -> dict:
    return {f"{k}{suffix}": v for k, v in d.items()}


def metrics_by_group(y_true: pd.Series, yhat_soft: pd.Series,
                     sensitive_features: pd.DataFrame, suffix: str = '',
                     threshold=0.5):
    # Check inputs
    assert isinstance(sensitive_features, pd.DataFrame)
    assert len(y_true) == len(yhat_soft)

    if len(y_true) <= 1: raise ValueError("Cannot compute metrics when n<=1.")

    if y_true.nunique() == 0:
        raise ValueError("only one unique label in y_true")

    yhat_hard = (yhat_soft >= threshold)
    if (suffix != '') and (not suffix.startswith('_')):
        # Ensure suffix has proper leading sep token
        suffix = '_' + suffix
    metrics = {}
    _log_loss = functools.partial(skm.log_loss, labels=[0., 1.])

    metric_fns_continuous = {
        'crossentropy': _log_loss,
        'cvar_doro': cvar_doro_metric,
        'cvar': cvar_metric,
        'loss_variance': loss_variance_metric,
    }

    metric_fns_binary = {
        'accuracy': skm.accuracy_score,
        'selection_rate': fairlearn.metrics.selection_rate,
        'count': fairlearn.metrics.count,
        'tpr': fairlearn.metrics.true_positive_rate,
        'fpr': fairlearn.metrics.false_positive_rate,
    }

    if all_subgroups_contain_all_label_values(y_true, sensitive_features):
        # Only compute AUC if all labels exist in each sens group. This is
        # due to a limitation in fairlearn.MetricFrame, which can't handle
        # errors or nan values when computing group difference metrics.
        metric_fns_binary['auc'] = skm.roc_auc_score
    else:
        logging.info("Not computing AUC for this split because one or more"
                     " sensitive subgroups do not contain all classes.")

    grouped_metrics_binary = fairlearn.metrics.MetricFrame(
        metrics=metric_fns_binary,
        y_true=y_true,
        y_pred=yhat_hard,
        sensitive_features=sensitive_features)

    grouped_metrics_continuous = fairlearn.metrics.MetricFrame(
        metrics=metric_fns_continuous,
        y_true=y_true,
        y_pred=yhat_soft,
        sensitive_features=sensitive_features)

    metrics.update(append_suffix_to_keys(
        grouped_metrics_binary.overall.to_dict(), suffix))

    metrics.update(append_suffix_to_keys(
        grouped_metrics_continuous.overall.to_dict(), suffix))

    for metric, value in grouped_metrics_continuous.difference().iteritems():
        metrics[f"abs_{metric}_disparity{suffix}"] = value

    # Compute some specific metrics of interest from the results
    metrics['abs_accuracy_disparity' + suffix] = \
        grouped_metrics_binary.difference()['accuracy']
    metrics['demographic_parity_diff' + suffix] = \
        grouped_metrics_binary.difference()['selection_rate']
    # EO diff is defined as  The greater of two metrics:
    # `true_positive_rate_difference` and `false_positive_rate_difference`;
    # see fairlearn.metrics.equalized_odds_difference
    metrics['equalized_odds_diff' + suffix] = \
        max(grouped_metrics_binary.difference()[['tpr', 'fpr']])

    metrics["accuracy_worstgroup" + suffix] = \
        grouped_metrics_binary.group_min()['accuracy']
    metrics["crossentropy_worstgroup" + suffix] = \
        grouped_metrics_continuous.group_max()['crossentropy']
    metrics['cvar_doro_worstgroup' + suffix] = \
        grouped_metrics_continuous.group_max()['cvar_doro']
    metrics['cvar_worstgroup' + suffix] = \
        grouped_metrics_continuous.group_max()['cvar']

    metrics = _intersectional_metrics_from_grouped_metrics(
        grouped_metrics_binary, metrics,
        suffix, sensitive_features)
    metrics = _intersectional_metrics_from_grouped_metrics(
        grouped_metrics_continuous, metrics,
        suffix, sensitive_features)

    return metrics

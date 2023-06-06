import logging
import os
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Optional, Mapping, Union, Callable, Any, Dict

import numpy as np
import torch
from ray.air import session
from ray.air.checkpoint import Checkpoint
from torch import nn
from torch.utils.data import DataLoader

from tableshift.models.optimizers import get_optimizer
from tableshift.models.torchutils import evaluate

OPTIMIZER_ARGS = ("lr", "weight_decay")


def append_by_key(from_dict: dict, to_dict: Union[dict, defaultdict]) -> dict:
    for k, v in from_dict.items():
        assert (k in to_dict) or (isinstance(to_dict, defaultdict))
        to_dict[k].append(v)
    return to_dict


class SklearnStylePytorchModel(ABC, nn.Module):
    """A pytorch model with an sklearn-style interface."""

    def __init__(self):
        super().__init__()

        # Indicator for domain generalization model
        self.domain_generalization = False

        # Indicator for domain adaptation model
        self.domain_adaptation = False

    def _init_optimizer(self):
        """(re)initialize the optimizer."""
        opt_config = {k: self.config[k] for k in OPTIMIZER_ARGS}
        logging.debug(f"initializing optimizer with params {opt_config}")
        self.optimizer = get_optimizer(self, config=opt_config)

    def predict(self, X) -> np.ndarray:
        """sklearn-compatible prediction function."""
        return self(X).detach().cpu().numpy()

    @abstractmethod
    def predict_proba(self, X) -> np.ndarray:
        """sklearn-compatible probability prediction function."""
        raise

    def evaluate(self, eval_loaders: Dict[str, DataLoader], device):
        return {str(split): evaluate(self, loader, device)
                for split, loader in eval_loaders.items()}

    @abstractmethod
    def train_epoch(self,
                    train_loaders: Dict[Any, DataLoader],
                    loss_fn: Callable,
                    device: str,
                    uda_loader: Optional[DataLoader] = None,
                    eval_loaders: Optional[Mapping[str, DataLoader]] = None,
                    # Terminate after this many steps if reached before end
                    # of epoch.
                    max_examples_per_epoch: Optional[int] = None
                    ) -> float:
        """Conduct one epoch of training and return the loss."""
        raise

    def save_checkpoint(self) -> Checkpoint:
        # Here we save a checkpoint. It is automatically registered with
        # Ray Tune and can be accessed through `session.get_checkpoint()`
        # API in future iterations.
        os.makedirs("model", exist_ok=True)
        torch.save(
            (self.state_dict(), self.optimizer.state_dict()),
            "model/checkpoint.pt")
        checkpoint = Checkpoint.from_directory("model")
        return checkpoint

    def fit(self, train_loaders: Dict[Any, DataLoader],
            loss_fn,
            device: str,
            n_epochs=1,
            eval_loaders: Optional[Dict[str, DataLoader]] = None,
            tune_report_split: Optional[str] = None,
            max_examples_per_epoch: Optional[int] = None) -> dict:
        fit_metrics = defaultdict(list)

        if tune_report_split:
            assert tune_report_split in list(eval_loaders.keys()) + ["train"]

        for epoch in range(1, n_epochs + 1):
            self.train_epoch(train_loaders=train_loaders,
                             loss_fn=loss_fn,
                             eval_loaders=eval_loaders,
                             device=device,
                             max_examples_per_epoch=max_examples_per_epoch)
            metrics = self.evaluate(eval_loaders, device=device)
            log_str = f'Epoch {epoch:03d} ' + ' | '.join(
                f"{k} score: {v:.4f}" for k, v in metrics.items())
            logging.info(log_str)

            checkpoint = self.save_checkpoint()

            if tune_report_split:
                session.report({"metric": metrics[tune_report_split]},
                               checkpoint=checkpoint)

            fit_metrics = append_by_key(from_dict=metrics, to_dict=fit_metrics)

        return fit_metrics


DOMAIN_GENERALIZATION_MODEL_NAMES = ["dann", "deepcoral", "irm", "mixup", "mmd",
                                     "vrex"]
DOMAIN_ADAPTATION_MODEL_NAMES = []
DOMAIN_ROBUSTNESS_MODEL_NAMES = ["group_dro", "dro"]
LABEL_ROBUSTNESS_MODEL_NAMES = ["aldro", "label_group_dro"]
SKLEARN_MODEL_NAMES = ("expgrad", "histgbm", "lightgbm", "wcs", "xgb")
BASELINE_MODEL_NAMES = ["ft_transformer", "mlp", "resnet", "node", "saint",
                        "tabtransformer"]
PYTORCH_MODEL_NAMES = BASELINE_MODEL_NAMES \
                      + DOMAIN_ROBUSTNESS_MODEL_NAMES \
                      + DOMAIN_GENERALIZATION_MODEL_NAMES \
                      + DOMAIN_ADAPTATION_MODEL_NAMES \
                      + LABEL_ROBUSTNESS_MODEL_NAMES


def is_domain_generalization_model_name(model_name: str) -> bool:
    return model_name in DOMAIN_GENERALIZATION_MODEL_NAMES


def is_domain_adaptation_model_name(model_name: str) -> bool:
    return model_name in DOMAIN_ADAPTATION_MODEL_NAMES


def is_pytorch_model_name(model: str) -> bool:
    """Helper function to determine whether a model name is a pytorch model.

    ISee description of is_pytorch_model() above."""
    is_sklearn = model in SKLEARN_MODEL_NAMES
    is_pt = model in PYTORCH_MODEL_NAMES
    assert is_sklearn or is_pt, f"unknown model name {model}"
    return is_pt

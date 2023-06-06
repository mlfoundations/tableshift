import logging
from typing import Any, Dict, Optional

from frozendict import frozendict
from ray.air import session
import torch
from tqdm import tqdm

from tableshift.core import TabularDataset
from tableshift.models.compat import SklearnStylePytorchModel, \
    is_domain_adaptation_model_name, is_domain_generalization_model_name
from tableshift.models.expgrad import ExponentiatedGradient
from tableshift.models.wcs import WeightedCovariateShiftClassifier
from tableshift.models.torchutils import unpack_batch, apply_model
from tableshift.models.losses import DomainLoss, GroupDROLoss
from tableshift.models.torchutils import get_module_attr

PYTORCH_DEFAULTS = frozendict({
    "lr": 0.001,
    "weight_decay": 0.0,
    "n_epochs": 1,
    "batch_size": 512,
})


def train_epoch(model, optimizer, criterion, train_loader,
                device) -> float:
    """Run one epoch of training, and return the training loss."""

    model.train()
    running_loss = 0.0
    n_train = 0
    n_batch = 0
    model_name = model.__class__.__name__
    for i, batch in tqdm(enumerate(train_loader), desc=f"{model_name}:train"):
        # get the inputs and labels
        inputs, labels, _, domains = unpack_batch(batch)
        inputs = inputs.float().to(device)
        labels = labels.float().to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = apply_model(model, inputs)
        if outputs.ndim == 2:
            outputs = outputs.squeeze(1)

        if isinstance(criterion, GroupDROLoss):
            # Case: loss requires domain labels, plus group weights + step size.
            domains = domains.float().to(device)
            group_weights = get_module_attr(model, "group_weights").to(device)
            group_weights_step_size = get_module_attr(
                model, "group_weights_step_size").to(device)
            loss = criterion(
                outputs, labels, domains,
                group_weights=group_weights,
                group_weights_step_size=group_weights_step_size,
                device=device)

        elif isinstance(criterion, DomainLoss):
            # Case: loss requires domain labels.
            domains = domains.float()
            loss = criterion(outputs, labels, domains)

        else:
            # Case: standard loss; only requires targets and predictions.
            loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        n_train += len(inputs)
        n_batch += 1
    logging.info(f"{model_name}:train: completed {n_batch} batches")
    return running_loss / n_train


def get_train_loaders(
        dset: TabularDataset,
        batch_size: int,
        model_name: Optional[str] = None,
        estimator: Optional[SklearnStylePytorchModel] = None,
) -> Dict[Any, torch.utils.data.DataLoader]:
    assert (model_name or estimator) and not (model_name and estimator), \
        "provide either model_name or estimator, but not both."
    if estimator.domain_generalization or is_domain_generalization_model_name(
            model_name):
        train_loaders = dset.get_domain_dataloaders("train", batch_size)
    elif estimator.domain_adaptation or is_domain_adaptation_model_name(
            model_name):
        raise NotImplementedError
    else:
        train_loaders = {"train": dset.get_dataloader("train", batch_size)}
    return train_loaders


def get_eval_loaders(
        dset: TabularDataset,
        batch_size: int,
        model_name: Optional[str] = None,
        estimator: Optional[SklearnStylePytorchModel] = None,
) -> Dict[Any, torch.utils.data.DataLoader]:
    assert (model_name or estimator) and not (model_name and estimator), \
        "provide either model_name or estimator, but not both."
    eval_loaders = {s: dset.get_dataloader(s, batch_size) for s in
                    dset.eval_split_names}
    if estimator.domain_generalization or is_domain_generalization_model_name(
            model_name):
        train_eval_loaders = dset.get_domain_dataloaders("train", batch_size,
                                                         infinite=False)
        eval_loaders.update(train_eval_loaders)
    elif estimator.domain_adaptation or is_domain_adaptation_model_name(
            model_name):
        raise NotImplementedError
    else:
        eval_loaders["train"] = dset.get_dataloader("train", batch_size)
    return eval_loaders


def _train_pytorch(estimator: SklearnStylePytorchModel, dset: TabularDataset,
                   device: str,
                   config=PYTORCH_DEFAULTS,
                   tune_report_split: str = None):
    """Helper function to train a pytorch estimator."""
    logging.debug(f"config is {config}")
    logging.debug(f"estimator is of type {type(estimator)}")
    logging.debug(f"dset name is {dset.name}")
    logging.debug(f"device is {device}")
    logging.debug(f"tune_report_split is {tune_report_split}")

    batch_size = config["batch_size"]
    train_loaders = get_train_loaders(estimator=estimator,
                                      dset=dset, batch_size=batch_size)
    eval_loaders = get_eval_loaders(estimator=estimator,
                                    dset=dset, batch_size=batch_size)

    loss_fn = config["criterion"]

    estimator.to(device)

    estimator.fit(train_loaders, loss_fn,
                  n_epochs=config["n_epochs"],
                  device=device,
                  eval_loaders=eval_loaders,
                  tune_report_split=tune_report_split,
                  max_examples_per_epoch=dset.n_train)
    return estimator


def _train_sklearn(estimator, dset: TabularDataset,
                   tune_report_split: str = None):
    """Helper function to train a sklearn-type estimator."""
    X_tr, y_tr, _, d_tr = dset.get_pandas(split="train")
    if isinstance(estimator, ExponentiatedGradient):
        estimator.fit(X_tr, y_tr, d=d_tr)
    elif isinstance(estimator, WeightedCovariateShiftClassifier):
        X_ood_tr, y_ood_tr, _, _ = dset.get_pandas(split="ood_validation")
        estimator.fit(X_tr, y_tr, X_ood_tr)
    else:
        estimator.fit(X_tr, y_tr)
    logging.info("fitting estimator complete.")

    if tune_report_split:
        X_te, _, _, _ = dset.get_pandas(split=tune_report_split)
        y_hat_te = estimator.predict(X_te)
        metrics = dset.evaluate_predictions(y_hat_te, split=tune_report_split)
        session.report({"metric": metrics[f"accuracy_{tune_report_split}"]})
    return estimator


def train(estimator: Any, dset: TabularDataset, tune_report_split: str = None,
          **kwargs):
    logging.info(f"fitting estimator of type {type(estimator)}")
    if isinstance(estimator, torch.nn.Module):
        assert isinstance(
            estimator,
            SklearnStylePytorchModel), \
            f"train() can only be called with SklearnStylePytorchModel; got " \
            f"type {type(estimator)} "
        return _train_pytorch(estimator, dset,
                              tune_report_split=tune_report_split, **kwargs)
    else:
        return _train_sklearn(estimator, dset,
                              tune_report_split=tune_report_split)

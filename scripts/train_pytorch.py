"""
A helper script to train a pytorch model without Ray. Useful for debugging.

Usage:
    python scripts/train_pytorch.py --model mlp --experiment adult
"""
import argparse

import numpy as np
import torch
import torchinfo
from sklearn.metrics import accuracy_score

from tableshift.core import get_dataset
from tableshift.models.default_hparams import get_default_config
from tableshift.models.torchutils import get_predictions_and_labels
from tableshift.models.training import train
from tableshift.models.utils import get_estimator


def main(experiment, cache_dir, model, debug: bool, use_cached: bool):
    if debug:
        print("[INFO] running in debug mode.")
        experiment = "_debug"

    dset = get_dataset(name=experiment, cache_dir=cache_dir,
                       use_cached=use_cached)
    config = get_default_config(model, dset)
    estimator = get_estimator(model, **config)
    print(torchinfo.summary(estimator))
    device = f"cuda:{torch.cuda.current_device()}" \
        if torch.cuda.is_available() else "cpu"

    print(f"device is {device}")
    train(estimator, dset, device=device, config=config)

    splits = ("id_test", "ood_test") if dset.is_domain_split else ("test",)
    for split in splits:
        loader = dset.get_dataloader(split)
        preds, labels = get_predictions_and_labels(estimator, loader, device)
        acc = accuracy_score(labels, np.round(preds))
        print(f'accuracy on split {split}: {acc:.4f}')
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", default="tmp",
                        help="Directory to cache raw data files to.")
    parser.add_argument("--debug", action="store_true", default=False,
                        help="Whether to run in debug mode. If True, various "
                             "truncations/simplifications are performed to "
                             "speed up experiment.")
    parser.add_argument("--experiment",
                        help="Experiment to run. Overridden when debug=True.")
    parser.add_argument("--model", default="mlp",
                        help="model to use.")
    parser.add_argument("--use_cached", default=False, action="store_true",
                        help="whether to use cached data.")
    args = parser.parse_args()
    main(**vars(args))

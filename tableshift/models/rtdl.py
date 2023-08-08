"""
Wrappers for tabular baseline models from the rtdl package.

rtdl source: https://github.com/Yura52/rtdl
"""

import copy
from typing import Optional, Callable, Any, Dict, Tuple

import numpy as np
import rtdl
from rtdl import FTTransformer, FeatureTokenizer, Transformer
import scipy
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from tableshift.models.compat import SklearnStylePytorchModel, OPTIMIZER_ARGS
from tableshift.models.torchutils import apply_model, get_module_attr
from tableshift.models.training import train_epoch

class WrappedFTTransformer(FTTransformer):
    @classmethod
    def _make(
            cls,
            n_num_features,
            cat_cardinalities,
            transformer_config,
    ):
        feature_tokenizer = FeatureTokenizer(
            n_num_features=n_num_features,
            cat_cardinalities=cat_cardinalities,
            d_token=transformer_config['d_token'],
        )
        if transformer_config['d_out'] is None:
            transformer_config['head_activation'] = None
        if transformer_config['kv_compression_ratio'] is not None:
            transformer_config['n_tokens'] = feature_tokenizer.n_tokens + 1
        return cls(
            feature_tokenizer,
            Transformer(**transformer_config),
        )


class SklearnStyleRTDLModel(SklearnStylePytorchModel):

    def train_epoch(self,
                    train_loaders: Dict[Any, DataLoader],
                    loss_fn: Callable,
                    device: str,
                    uda_loader: Optional[DataLoader] = None,
                    eval_loaders: Optional[Dict[str, DataLoader]] = None,
                    max_examples_per_epoch: Optional[int] = None
                    ) -> float:
        """Run a single epoch of model training."""
        assert len(train_loaders.values()) == 1
        train_loader = list(train_loaders.values())[0]
        return train_epoch(self, self.optimizer, loss_fn, train_loader, device)

    @torch.no_grad()
    def predict_proba(self, X):
        prediction = self.predict(X)
        return scipy.special.expit(prediction)


class ResNetModel(rtdl.ResNet, SklearnStyleRTDLModel):
    def __init__(self, **hparams):
        self.config = copy.deepcopy(hparams)

        # Remove hparams that are not taken by the rtdl constructor.
        for k in OPTIMIZER_ARGS:
            hparams.pop(k)

        super().__init__(**hparams)
        self._init_optimizer()

    def predict_proba(self, X) -> np.ndarray:
        return self.predict_proba(X)


class MLPModel(rtdl.MLP, SklearnStyleRTDLModel):
    def __init__(self, **hparams):
        self.config = copy.deepcopy(hparams)

        # Remove hparams that are not taken by the rtdl constructor.
        for k in OPTIMIZER_ARGS:
            hparams.pop(k)

        super().__init__(**hparams)
        self._init_optimizer()

    def predict_proba(self, X) -> np.ndarray:
        return self.predict_proba(X)


class MLPModelWithHook(MLPModel):
    def get_activations(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward method with hook."""
        # Set up feature extraction hooks

        # Use the pre-activation, pre-dropout output of the linear layer of final block
        block_num = len(get_module_attr(self, "blocks")) - 1
        layer = "linear"
        # The key used to find the activations in the dictionary.
        activations_key = f'block{block_num}{layer}'

        activation = {}

        def get_activation():
            """Utility function to fetch an activation."""

            def hook(self, input, output):
                activation[activations_key] = output.detach()

            return hook

        if hasattr(self, "module"):
            # Case: distributed module; access the module explicitly.
            self.module.blocks[block_num].linear.register_forward_hook(
                get_activation())
        else:  # Case: standard module.
            self.blocks[block_num].linear.register_forward_hook(
                get_activation())

        def _get_activations(inputs) -> Tuple[Tensor, Tensor]:
            """Apply model and return the (outputs,activations) tuple."""
            outputs = apply_model(self, inputs).squeeze(1)
            activations = activation[activations_key]
            return activations

        return _get_activations(x)


class FTTransformerModel(WrappedFTTransformer, SklearnStyleRTDLModel):

    @property
    def cat_idxs(self):
        return getattr(self.feature_tokenizer.cat_tokenizer,
                       "cat_cardinalities", [])

    def predict_proba(self, X) -> np.ndarray:
        return self.predict_proba(X)

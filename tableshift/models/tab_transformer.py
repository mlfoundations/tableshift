import copy
from typing import Optional, Callable, Any, Dict

import scipy
import torch
from tab_transformer_pytorch import TabTransformer
from torch.utils.data import DataLoader

from tableshift.models.compat import SklearnStylePytorchModel, OPTIMIZER_ARGS
from tableshift.models.training import train_epoch


class TabTransformerModel(TabTransformer, SklearnStylePytorchModel):
    def __init__(self, **hparams):
        self.config = copy.deepcopy(hparams)

        # Remove hparams that are not taken by the rtdl constructor.
        for k in OPTIMIZER_ARGS:
            hparams.pop(k)

        self.cat_idxs = hparams.pop("cat_idxs")

        super().__init__(**hparams)
        self._init_optimizer()

    def forward(self, x_cont: torch.Tensor, x_categ: Optional[torch.Tensor]):
        """Forward pass with argument values reversed (to match other models).

        Also handles empty categorical values."""
        if x_categ is None:
            x_categ = torch.Tensor([])
        else:
            x_categ = x_categ.long()
        return TabTransformer.forward(self, x_categ=x_categ, x_cont=x_cont)

    @torch.no_grad()
    def predict_proba(self, X):
        prediction = self.predict(X)
        return scipy.special.expit(prediction)

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

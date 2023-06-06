from typing import Optional, Callable, Any, Dict

import scipy
import torch
from torch.utils.data import DataLoader

from tableshift.models.compat import SklearnStylePytorchModel
from tableshift.models.training import train_epoch
from tableshift.third_party import node


class NodeModel(SklearnStylePytorchModel):
    def __init__(self, d_in: int,
                 depth: int,
                 num_layers: int,
                 total_tree_count: int,
                 tree_dim: int,
                 choice_function=node.entmax15,
                 bin_function=node.entmoid15, **hparams):

        self.config = hparams

        super().__init__()

        self.d_in = d_in
        self.num_layers = num_layers
        self.tree_dim = tree_dim
        self.depth = depth
        self.choice_function = choice_function
        self.bin_function = bin_function
        self.total_tree_count = total_tree_count
        self.layers = torch.nn.ModuleList()
        self._init_layers()
        self._init_optimizer()

    @property
    def trees_per_layer(self):
        return int(self.total_tree_count / self.num_layers)

    def _init_layers(self):
        self.layers = torch.nn.ModuleList()
        self.layers.append(
            node.DenseBlock(self.d_in, self.trees_per_layer,
                            num_layers=self.num_layers,
                            tree_dim=self.tree_dim,
                            depth=self.depth,
                            flatten_output=False,
                            choice_function=self.choice_function,
                            bin_function=self.bin_function))
        # Lambda layer averages first channels of every tree
        self.layers.append(node.Lambda(lambda x: x[..., 0].mean(dim=-1)))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        # as in NODE training code, we do not use activation functions
        # between layers nor at output layer
        # https://github.com/Qwicen/node/blob/3bae6a8a63f0205683270b6d566d9cfa659403e4/lib/trainer.py#LL115C1-L123C71
        return x

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

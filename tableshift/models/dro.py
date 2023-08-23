from typing import Optional, Mapping, Dict, Any, Callable

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from tableshift.models.compat import SklearnStylePytorchModel
from tableshift.models.rtdl import MLPModel, ResNetModel, FTTransformerModel
from tableshift.models.torchutils import unpack_batch, apply_model


class GroupDROModel(MLPModel, SklearnStylePytorchModel):
    def __init__(self, group_weights_step_size: float, n_groups: int, **kwargs):
        MLPModel.__init__(self, **kwargs)

        assert n_groups > 0, "require nonzero n_groups."
        self.group_weights_step_size = torch.Tensor([group_weights_step_size])
        # initialize adversarial weights
        self.group_weights = torch.nn.Parameter(
            torch.full([n_groups], 1. / n_groups))

    def to(self, device):
        super().to(device)
        for attr in ("group_weights_step_size", "group_weights"):
            setattr(self, attr, getattr(self, attr).to(device))
        return self

    def train_epoch(self,
                    train_loaders: Dict[Any, DataLoader],
                    loss_fn: Callable,
                    device: str,
                    uda_loader: Optional[DataLoader] = None,
                    eval_loaders: Optional[Dict[str, DataLoader]] = None,
                    max_examples_per_epoch: Optional[int] = None
                    ) -> float:
        raise


class LabelGroupDROModel(GroupDROModel):
    """Group DRO with class labels as groups. (For class/label robustness.)"""

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
        assert len(train_loaders.values()) == 1
        train_loader = list(train_loaders.values())[0]

        for iteration, batch in tqdm(enumerate(train_loader),
                                     desc="groupdro:train"):
            x_batch, y_batch, _, _ = unpack_batch(batch)
            x_batch = x_batch.float().to(device)
            y_batch = y_batch.float().to(device)
            self.train()
            self.optimizer.zero_grad()
            outputs = apply_model(self, x_batch)
            loss = loss_fn(outputs.squeeze(1), y_batch, y_batch,
                           self.group_weights,
                           self.group_weights_step_size)

            loss.backward()
            self.optimizer.step()


class DomainGroupDROResNetModel(ResNetModel, SklearnStylePytorchModel):
    """Group DRO with domain labels as groups. (For domain robustness.)"""

    def __init__(self, group_weights_step_size: float, n_groups: int, **kwargs):
        ResNetModel.__init__(self, **kwargs)
        assert n_groups > 0, "require nonzero n_groups."
        self.group_weights_step_size = torch.Tensor([group_weights_step_size])
        # initialize adversarial weights
        self.group_weights = torch.nn.Parameter(
            torch.full([n_groups], 1. / n_groups))

    def to(self, device):
        super().to(device)
        for attr in ("group_weights_step_size", "group_weights"):
            setattr(self, attr, getattr(self, attr).to(device))
        return self

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
        assert len(train_loaders.values()) == 1
        train_loader = list(train_loaders.values())[0]

        for iteration, batch in tqdm(enumerate(train_loader),
                                     desc="groupdro-resnet:train"):
            x_batch, y_batch, _, d_batch = unpack_batch(batch)
            x_batch = x_batch.float().to(device)
            y_batch = y_batch.float().to(device)
            d_batch = d_batch.float().to(device)
            self.train()
            self.optimizer.zero_grad()
            outputs = apply_model(self, x_batch)
            loss = loss_fn(outputs.squeeze(1), y_batch, d_batch,
                           self.group_weights,
                           self.group_weights_step_size)

            loss.backward()
            self.optimizer.step()

class DomainGroupDROFTTransformerModel(FTTransformerModel, SklearnStylePytorchModel):
    """Group DRO with domain labels as groups. (For domain robustness.)"""
    def to(self, device):
        super().to(device)
        for attr in ("group_weights_step_size", "group_weights"):
            setattr(self, attr, getattr(self, attr).to(device))
        return self

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
        assert len(train_loaders.values()) == 1
        train_loader = list(train_loaders.values())[0]

        for iteration, batch in tqdm(enumerate(train_loader),
                                     desc="groupdro-fttransformer:train"):
            x_batch, y_batch, _, d_batch = unpack_batch(batch)
            x_batch = x_batch.float().to(device)
            y_batch = y_batch.float().to(device)
            d_batch = d_batch.float().to(device)
            self.train()
            self.optimizer.zero_grad()
            outputs = apply_model(self, x_batch)
            loss = loss_fn(outputs.squeeze(1), y_batch, d_batch,
                           self.group_weights,
                           self.group_weights_step_size)

            loss.backward()
            self.optimizer.step()


class DomainGroupDROModel(GroupDROModel):
    """Group DRO with domain labels as groups. (For domain robustness.)"""

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
        assert len(train_loaders.values()) == 1
        train_loader = list(train_loaders.values())[0]

        for iteration, batch in tqdm(enumerate(train_loader),
                                     desc="groupdro:train"):
            x_batch, y_batch, _, d_batch = unpack_batch(batch)
            x_batch = x_batch.float().to(device)
            y_batch = y_batch.float().to(device)
            d_batch = d_batch.float().to(device)
            self.train()
            self.optimizer.zero_grad()
            outputs = apply_model(self, x_batch)
            loss = loss_fn(outputs.squeeze(1), y_batch, d_batch,
                           self.group_weights,
                           self.group_weights_step_size)

            loss.backward()
            self.optimizer.step()


class AdversarialLabelDROModel(MLPModel, SklearnStylePytorchModel):
    """Implements model from `Coping with label shift via distributionally
    robust optimization`, Zhang et al ICLR 2021.
    See https://arxiv.org/pdf/2010.12230.pdf .

    Some implementation ideas from https://github.com/ShahryarBQ/DRO .
    """

    def __init__(self,
                 n_groups: int,  # number of classes for y
                 eta_pi: float,  # learning rate for lagrangian dual variable
                 r: float,
                 clip_max: float = 2.,
                 eps: float = 0.001,
                 beta: float = 0.999,  # exponential moving average (see sec. B)
                 **kwargs):
        MLPModel.__init__(self, **kwargs)
        self.beta = beta
        self.clip_max = clip_max
        self.eps = eps
        self.eta_pi = eta_pi

        self.r = r
        self.p_emp = torch.full([n_groups], 1. / n_groups)  # p(y==0), p(y==1)

        self.pi_t = torch.nn.Parameter(
            torch.full([n_groups], 1. / n_groups), requires_grad=True)

    def to(self, device):
        super().to(device)
        for attr in ("pi_t",):
            setattr(self, attr, getattr(self, attr).to(device))
        return self

    def _extract_gradient_pi(self, X, y, criterion):

        if self.pi_t.grad is not None:
            self.pi_t.grad.zero_()
        pred = self(X)
        loss = criterion(- self.pi_t[y] * pred, y.float().unsqueeze(0))
        loss.backward()
        return self.pi_t.grad

    def train_epoch(self,
                    train_loaders: Dict[Any, DataLoader],
                    loss_fn: Callable,
                    device: str,
                    uda_loader: Optional[DataLoader] = None,
                    eval_loaders: Optional[Dict[str, DataLoader]] = None,
                    max_examples_per_epoch: Optional[int] = None
                    ) -> float:
        assert len(train_loaders.values()) == 1
        train_loader = list(train_loaders.values())[0]

        for iteration, batch in tqdm(enumerate(train_loader),
                                     desc=f"{self.__class__.__name__}:train"):
            x_batch, y_batch, _, _ = unpack_batch(batch)
            batch_size = len(x_batch)
            x_batch = x_batch.float().to(device)
            y_batch = y_batch.float().to(device)
            self.p_emp = self.p_emp.to(device)

            p_batch = y_batch.mean().item()
            self.p_emp = (
                    self.beta * self.p_emp +
                    (1 - self.beta) * torch.Tensor([1 - p_batch, p_batch]).to(
                device))

            self.train()
            self.optimizer.zero_grad()
            outputs = apply_model(self, x_batch)

            # Update model parameters

            # Step 4 of algorithm 1 in https://arxiv.org/pdf/2010.12230.pdf
            weights = (torch.where(y_batch == 1, self.pi_t[1], self.pi_t[0]) /
                       torch.where(y_batch == 1, self.p_emp[1], self.p_emp[0]))

            loss = loss_fn(outputs.squeeze(1), y_batch, reduction='none')
            loss = torch.dot(loss, weights)

            loss.backward()
            self.optimizer.step()

            # Update lagrangian variable
            g_pi = torch.zeros_like(self.pi_t)
            for X, y in zip(x_batch, y_batch.long()):
                sample_grad_pi = self._extract_gradient_pi(X, y, loss_fn)
                g_pi += (1 / batch_size) * (
                        1 / self.p_emp[y]) * sample_grad_pi
            g_pi = torch.clip(g_pi, max=self.clip_max)  # Section 3.5

            with torch.no_grad():
                KL = torch.nn.functional.kl_div(torch.log(self.pi_t),
                                                self.p_emp,
                                                reduction='batchmean')

                # set 2 * self.gamma * self.lambda_val = 1; cf. Sec. B
                alpha = 0 if self.r > KL else 1

                C = torch.norm(
                    (self.pi_t * self.p_emp ** alpha) ** (1 / (1 + alpha))
                    * torch.exp(self.eta_pi * g_pi),
                    p=1)
                pi_t = (self.pi_t * (self.p_emp ** alpha)) ** (
                        1 / (1 + alpha)) * torch.exp(self.eta_pi * g_pi) / C
                pi_t += self.eps

            self.pi_t = torch.nn.Parameter(pi_t, requires_grad=True)

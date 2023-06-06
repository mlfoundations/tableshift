from typing import Tuple
import copy
import re

import torch
import torch.nn.functional as F
from torch import Tensor
import torch.autograd as autograd

from tableshift.models.rtdl import MLPModel
from tableshift.models.domain_generalization import DomainGeneralizationModel
from tableshift.models.torchutils import apply_model
from tableshift.models.torchutils import get_module_attr


class DANNModel(DomainGeneralizationModel):
    """Domain-Adversarial Neural Networks."""

    def __init__(self, d_steps_per_g_step: int, grad_penalty: float,
                 loss_lambda: float,
                 class_balance=False, **hparams):
        self.config = copy.deepcopy(hparams)
        self.class_balance = class_balance
        self.d_steps_per_g_step = d_steps_per_g_step
        self.loss_lambda = loss_lambda
        self.grad_penalty = grad_penalty

        # Initialize the main ("g") network
        g_hparams = {re.sub("_g$", "", k): v for k, v in hparams.items()
                     if not k.endswith("_d")}
        super().__init__(**g_hparams)

        # Initialize the discriminator network
        d_hparams = {re.sub("_d$", "", k): v for k, v in hparams.items()
                     if not k.endswith("_g")}
        d_hparams["d_in"] = g_hparams["d_layers"][-1]
        # import ipdb;ipdb.set_trace()

        self.discriminator = MLPModel(**d_hparams)

        self.register_buffer('update_count', torch.tensor([0]))

    @property
    def disc_opt(self) -> torch.optim.Optimizer:
        return self.discriminator.optimizer

    @property
    def gen_opt(self) -> torch.optim.Optimizer:
        return self.optimizer

    def update(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        self.update_count += 1
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])

        # Set up feature extraction hooks

        # Use the pre-activation, pre-dropout output of the linear layer of
        # final block
        block_num = len(get_module_attr(self, "blocks")) - 1
        layer = "linear"
        # The key used to find the activations in the dictionary.
        activations_key = f'block{block_num}{layer}'

        activation = {}

        def get_activation():
            """Utility function to fetch an activation."""

            def hook(self, input, output):
                activation[activations_key] = output

            return hook

        if hasattr(self, "module"):
            # Case: distributed module; access the module explicitly.
            self.module.blocks[block_num].linear.register_forward_hook(
                get_activation())
        else:  # Case: standard module.
            self.blocks[block_num].linear.register_forward_hook(
                get_activation())

        def _get_outputs_and_activations(inputs) -> Tuple[Tensor, Tensor]:
            """Apply model and return the (outputs,activations) tuple."""
            outputs = apply_model(self, inputs).squeeze(1)
            activations = activation[activations_key]
            return outputs, activations

        all_preds, all_z = _get_outputs_and_activations(all_x)

        disc_input = all_z

        disc_out = apply_model(self.discriminator, disc_input).squeeze(1)
        disc_labels = torch.cat([
            torch.full((x.shape[0],), i, dtype=torch.float, device=device)
            for i, (x, y) in enumerate(minibatches)
        ])

        if self.class_balance:
            y_counts = F.one_hot(all_y).sum(dim=0)
            weights = 1. / (y_counts[all_y] * y_counts.shape[0]).float()
            disc_loss = F.cross_entropy(disc_out, disc_labels, reduction='none')
            disc_loss = (weights * disc_loss).sum()
        else:
            disc_loss = F.cross_entropy(disc_out, disc_labels)

        input_grad = autograd.grad(
            F.cross_entropy(disc_out, disc_labels, reduction='sum'),
            [disc_input], create_graph=True)[0]
        grad_penalty = (input_grad ** 2).sum(dim=1).mean(dim=0)
        disc_loss += self.grad_penalty * grad_penalty

        d_steps_per_g = self.d_steps_per_g_step
        ce_loss = F.cross_entropy(all_preds, all_y)

        if (self.update_count.item() % (1 + d_steps_per_g) < d_steps_per_g):

            self.disc_opt.zero_grad()
            disc_loss.backward()
            self.disc_opt.step()
            return {'disc_loss': disc_loss.item(), 'loss': ce_loss.item()}
        else:
            # all_preds = self.classifier(all_z)
            gen_loss = (ce_loss +
                        (self.loss_lambda * -disc_loss))
            self.disc_opt.zero_grad()
            self.gen_opt.zero_grad()
            gen_loss.backward()
            self.gen_opt.step()
            return {'gen_loss': gen_loss.item(), 'loss': ce_loss.item()}

    def predict(self, x):
        return apply_model(self, x)

    def to(self, device):
        super().to(device)
        self.discriminator = self.discriminator.to(device)
        return self

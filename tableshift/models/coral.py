import copy
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from tableshift.models.domain_generalization import DomainGeneralizationModel
from tableshift.models.torchutils import apply_model
from tableshift.models.torchutils import get_module_attr


class AbstractMMD(DomainGeneralizationModel):
    """
    Perform ERM while matching the pair-wise domain feature distributions
    using MMD (abstract class).

    Adapted from DomainBench.algorithms.AbstractMMD.
    """

    def __init__(self, gaussian: bool, **hparams):
        self.config = copy.deepcopy(hparams)

        super().__init__(**hparams)
        if gaussian:
            self.kernel_type = "gaussian"
        else:
            self.kernel_type = "mean_cov"

    def my_cdist(self, x1, x2):
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        res = torch.addmm(x2_norm.transpose(-2, -1),
                          x1,
                          x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
        return res.clamp_min_(1e-30)

    def gaussian_kernel(self, x, y, gamma=[0.001, 0.01, 0.1, 1, 10, 100,
                                           1000]):
        D = self.my_cdist(x, y)
        K = torch.zeros_like(D)

        for g in gamma:
            K.add_(torch.exp(D.mul(-g)))

        return K

    def mmd(self, x, y):
        if self.kernel_type == "gaussian":
            Kxx = self.gaussian_kernel(x, x).mean()
            Kyy = self.gaussian_kernel(y, y).mean()
            Kxy = self.gaussian_kernel(x, y).mean()
            return Kxx + Kyy - 2 * Kxy
        else:
            mean_x = x.mean(0, keepdim=True)
            mean_y = y.mean(0, keepdim=True)
            cent_x = x - mean_x
            cent_y = y - mean_y
            cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
            cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

            mean_diff = (mean_x - mean_y).pow(2).mean()
            cova_diff = (cova_x - cova_y).pow(2).mean()

            return mean_diff + cova_diff

    def update(self, minibatches, unlabeled=None):
        objective = 0
        penalty = 0
        nmb = len(minibatches)

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

        def _get_outputs_and_activations(inputs) -> Tuple[Tensor, Tensor]:
            """Apply model and return the (outputs,activations) tuple."""
            outputs = apply_model(self, inputs).squeeze(1)
            activations = activation[activations_key]
            return outputs, activations

        outputs_and_activations = [_get_outputs_and_activations(x) for x, _ in
                                   minibatches]

        outputs = [x[0] for x in outputs_and_activations]
        features = [x[1] for x in outputs_and_activations]
        targets = [yi for _, yi in minibatches]

        for i in range(nmb):
            objective += F.cross_entropy(outputs[i], targets[i])
            for j in range(i + 1, nmb):
                penalty += self.mmd(features[i], features[j])

        objective /= nmb
        if nmb > 1:
            penalty /= (nmb * (nmb - 1) / 2)

        self.optimizer.zero_grad()
        (objective + (self.mmd_gamma * penalty)).backward()
        self.optimizer.step()

        if torch.is_tensor(penalty):
            penalty = penalty.item()

        return {'loss': objective.item(), 'penalty': penalty}


class MMDModel(AbstractMMD):
    """
    MMD using Gaussian kernel; via DomainBench.
    """

    def __init__(self, mmd_gamma: float, gaussian=True, **hparams):
        self.mmd_gamma = mmd_gamma
        super().__init__(**hparams, gaussian=gaussian)


class DeepCoralModel(AbstractMMD):
    """
    MMD using mean and covariance difference; via DomainBench.
    """

    def __init__(self, mmd_gamma: float, gaussian=False, **hparams):
        self.mmd_gamma = mmd_gamma
        super().__init__(**hparams, gaussian=gaussian)

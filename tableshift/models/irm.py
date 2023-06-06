import copy

import torch
import torch.nn.functional as F
import torch.autograd as autograd

from tableshift.models.domain_generalization import DomainGeneralizationModel
from tableshift.models.torchutils import apply_model


class IRMModel(DomainGeneralizationModel):
    """Class to represent Invariant Risk Minimization models.

        Based on implementation from
        https://github.com/facebookresearch/DomainBed/blob/main/domainbed
        /algorithms.py .
        """

    def __init__(self, irm_lambda: float, irm_penalty_anneal_iters: int,
                 **hparams):
        self.config = copy.deepcopy(hparams)

        super().__init__(**hparams)

        self.irm_lambda = irm_lambda
        self.irm_penalty_anneal_iters = irm_penalty_anneal_iters
        self.register_buffer('update_count', torch.tensor([0]))

    @staticmethod
    def _irm_penalty(logits, y):
        device = "cuda" if logits.is_cuda else "cpu"
        scale = torch.tensor(1.).to(device).requires_grad_()
        loss_1 = F.cross_entropy(logits[::2] * scale, y[::2])
        loss_2 = F.cross_entropy(logits[1::2] * scale, y[1::2])
        grad_1 = autograd.grad(loss_1, [scale], create_graph=True)[0]
        grad_2 = autograd.grad(loss_2, [scale], create_graph=True)[0]
        result = torch.sum(grad_1 * grad_2)
        return result

    def update(self, minibatches, unlabeled=None):

        penalty_weight = (
            self.irm_lambda
            if self.update_count >= self.irm_penalty_anneal_iters
            else 1.0)
        nll = 0.
        penalty = 0.
        all_x = torch.cat([x for x, y in minibatches])
        all_logits = apply_model(self, all_x).squeeze()
        all_logits_idx = 0

        for i, (x, y) in enumerate(minibatches):
            logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            nll += F.cross_entropy(logits, y)
            penalty += self._irm_penalty(logits, y)

        nll /= len(minibatches)
        penalty /= len(minibatches)
        loss = nll + (penalty_weight * penalty)

        if self.update_count == self.irm_penalty_anneal_iters:
            # Reset optimizer (as in DomainBed), because Adam optimizer
            # doesn't like the sharp jump in gradient magnitudes that happens
            # at this step.
            self._init_optimizer()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        return {'loss': loss.item(), 'nll': nll.item(),
                'penalty': penalty.item()}

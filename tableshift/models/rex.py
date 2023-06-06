import copy

import torch
import torch.nn.functional as F

from tableshift.models.domain_generalization import DomainGeneralizationModel
from tableshift.models.torchutils import apply_model


class VRExModel(DomainGeneralizationModel):
    """V-REx algorithm from http://arxiv.org/abs/2003.00688, adapted from
    DomainBed."""

    def __init__(self, vrex_penalty_anneal_iters: int,
                 vrex_lambda: float, **hparams):
        self.config = copy.deepcopy(hparams)
        self.vrex_penalty_anneal_iters = vrex_penalty_anneal_iters
        self.vrex_lambda = vrex_lambda

        super().__init__(**hparams)
        self.register_buffer('update_count', torch.tensor([0]))

    def update(self, minibatches, unlabeled=None):
        if self.update_count >= self.vrex_penalty_anneal_iters:
            penalty_weight = self.vrex_lambda
        else:
            penalty_weight = 1.0

        nll = 0.

        all_x = torch.cat([x for x, y in minibatches])
        all_logits = apply_model(self, all_x).squeeze(1)

        all_logits_idx = 0
        losses = torch.zeros(len(minibatches))
        for i, (x, y) in enumerate(minibatches):
            logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            nll = F.cross_entropy(logits, y)
            losses[i] = nll

        mean = losses.mean()
        penalty = ((losses - mean) ** 2).mean()
        loss = mean + penalty_weight * penalty

        if self.update_count == self.vrex_penalty_anneal_iters:
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

import copy

import numpy as np
import torch
import torch.nn.functional as F
import torch.autograd as autograd

from tableshift.models.domain_generalization import DomainGeneralizationModel
from tableshift.models.torchutils import apply_model
from tableshift.third_party.domainbed import random_pairs_of_minibatches


class MixUpModel(DomainGeneralizationModel):
    """
    Mixup of minibatches from different domains
    https://arxiv.org/pdf/2001.00677.pdf
    https://arxiv.org/pdf/1912.01805.pdf
    """

    def __init__(self, **hparams):
        self.config = copy.deepcopy(hparams)
        self.mixup_alpha = hparams.pop("mixup_alpha")

        super().__init__(**hparams)

    def update(self, minibatches, unlabeled=None):
        objective = 0

        for (xi, yi), (xj, yj) in random_pairs_of_minibatches(minibatches):
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)

            x = lam * xi + (1 - lam) * xj
            predictions = apply_model(self, x).squeeze()

            objective += lam * F.cross_entropy(predictions, yi)
            objective += (1 - lam) * F.cross_entropy(predictions, yj)

        objective /= len(minibatches)

        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        return {'loss': objective.item()}

#
# class MixUpModel(MLPModel, SklearnStylePytorchModel):
#     """
#     Class to train via Mixup of batches from different domains.
#
#     Implementation via:
#         https://github.com/facebookresearch/DomainBed/blob/main/domainbed/algorithms.py#L413
#     Citations:
#         https://arxiv.org/pdf/2001.00677.pdf
#         https://arxiv.org/pdf/1912.01805.pdf
#     """
#
#     def __init__(self, mixup_alpha: float, **hparams):
#         self.config = copy.deepcopy(hparams)
#
#         super().__init__(**hparams)
#         self.mixup_alpha = mixup_alpha
#
#     def train_epoch(self, train_loaders: torch.utils.data.DataLoader,
#                     loss_fn: Callable,
#                     device: str,
#                     eval_loaders: Optional[
#                         Mapping[str, torch.utils.data.DataLoader]] = None,
#                     ) -> float:
#         total_loss = None
#         n_train = 0
#         microbatch_size = 16  # must evenly divide batch size
#
#         for batch in train_loaders:
#             loss = 0
#
#             x_batch, y_batch, _, _ = unpack_batch(batch)
#             batch_size = len(x_batch)
#
#             for idxs_i, idxs_j in random_minibatch_idxs(batch_size,
#                                                         microbatch_size):
#
#                 lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
#                 x = lam * x_batch[idxs_i] + (1 - lam) * x_batch[idxs_j]
#                 predictions = apply_model(self, x).squeeze()
#                 loss += lam * loss_fn(predictions, y_batch[idxs_i])
#                 loss += (1 - lam) * loss_fn(predictions, y_batch[idxs_j])
#
#             loss /= batch_size
#             self.optimizer.zero_grad()
#             loss.backward()
#             self.optimizer.step()
#
#             batch_loss = loss.detach().cpu().numpy().item()
#             n_train += batch_size
#
#             if total_loss is None:
#                 total_loss = batch_loss
#             else:
#                 total_loss += batch_loss
#
#         return total_loss / n_train

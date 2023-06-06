from abc import abstractmethod
import logging
from typing import Union, Dict, Any, Callable, Optional, Mapping, Tuple

from torch import Tensor
from torch.utils.data import DataLoader

from tableshift.models.rtdl import MLPModel
from tableshift.models.torchutils import unpack_batch


class DomainGeneralizationModel(MLPModel):
    """Class to represent models trained for domain generalization."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.register_buffer('examples_seen', Tensor([0.]), persistent=False)
        self.domain_generalization = True

    @abstractmethod
    def update(self, minibatches, unlabeled=None):
        raise

    def train_epoch(self,
                    train_loaders: Union[DataLoader, Dict[Any, DataLoader]],
                    loss_fn: Callable,
                    device: str,
                    uda_loader: Optional[DataLoader] = None,
                    eval_loaders: Optional[Mapping[str, DataLoader]] = None,
                    max_examples_per_epoch: Optional[int] = None
                    ) -> float:
        """Conduct one epoch of training and return the loss."""

        loaders = [x for x in train_loaders.values()]
        train_minibatches_iterator = zip(*loaders)

        def _prepare_batch(batch) -> Tuple[Tensor, Tensor]:
            x_batch, y_batch, _, _ = unpack_batch(batch)
            return x_batch.float().to(device), y_batch.float().to(device)

        loss = None
        self.examples_seen.zero_()
        while True:
            logging.info(f"{self.__class__.__name__}:train examples seen: "
                         f"{self.examples_seen.item()} of {max_examples_per_epoch}")
            batches = [_prepare_batch(batch) for batch in
                       next(train_minibatches_iterator)]
            # Note: if this was a domain_adaption task, do the same as above
            # for uda_loader.
            tmp = self.update(batches)

            loss = tmp['loss'] if loss is None else loss + tmp['loss']
            self.examples_seen += sum(len(batch_x) for batch_x, _ in batches)
            if self.examples_seen.item() >= max_examples_per_epoch:
                break

        return loss / self.examples_seen.item()

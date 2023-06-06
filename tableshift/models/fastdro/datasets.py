"""
Dataset objects for ImageNet and Digits, requires
features extracted from the python scripts in `features/`.

Code from https://github.com/daniellevy/fast-dro
"""
import torch
import numpy as np

from torchvision.datasets import ImageNet
from torch.utils.data import Sampler, Dataset, TensorDataset

from scipy.optimize import minimize

import os

import logging

from itertools import product as product_iter

from tableshift.models.fastdro import utils

from collections import Mapping


DATASETS = ['mnist_typed', 'imagenet']


class GroupedDataset(Dataset):
    def __init__(self, dataset: TensorDataset, group_ids, group_names):
        self.group_names = group_names
        self.group_ids = group_ids

        self.group_sizes = utils.aggregate_by_group(
            np.ones_like(group_ids), group_ids, len(group_names)).astype('int')

        self.dataset = dataset
        assert len(dataset) == len(group_ids)

        logging.info(f'Creating dataset with {len(group_names)} groups')
        if len(group_names) < 100:
            labels = np.array(dataset.tensors[1])
            for i in range(len(group_names)):
                group_hist = np.unique(labels[np.array(group_ids) == i],
                                       return_counts=True)
                logging.info(f'Label breakdown for group {i}={group_names[i]}: '
                             f'{tuple(zip(*group_hist))}; '
                             f'total={self.group_sizes[i]}')

    def __getitem__(self, index):
        return tuple(self.dataset[index]) + (self.group_ids[index],)

    def __len__(self):
        return len(self.dataset)

    @property
    def num_features(self):
        return self.dataset.tensors[0].shape[1]


class MNISTandTypedFeatures(GroupedDataset):
    """Dataset object with group structure for Digits experiment"""
    def __init__(self, root='data',
                 features_name='lenet',
                 train=True,
                 group_by_class=True,
                 subsample_to=(-1, -1),
                 subsample_seed=123):
        """

        Parameters
        ----------

        root : string
            Path to the root of the data folder
        features_names : string
            Features expected to be in {root}/[mnist or typed]/{features_names}
        train : bool
            If True, loads the train set
        group_by_class : bool:
            If True groups are of the form (population, label) otherwise
            just (population)
        subsample_to : (int, int)
            Number of examples to take from mnist and typed.
        subsample_seed:
            Seed that decides the randomness of the subsampling.

        """
        split = 'train' if train else 'val'
        if len(subsample_to) < 2:
            subsample_to = tuple(subsample_to) + (-1,) * (2 - len(subsample_to))

        if group_by_class:
            group_names = [f'{s}_{d}' for s in ('mnist', 'typed')
                           for d in range(10)]
        else:
            group_names = ['mnist', 'typed']

        features = []
        targets = []
        group_ids = []

        for i, name in enumerate(['MNIST', 'typed_digits']):
            features_path = os.path.join(root, name,
                                         features_name + '_features',
                                         split + '.pt')
            data = torch.load(features_path)
            xy = data['features'], data['targets']
            if subsample_to[i] > 0:
                xy = utils.subsample_arrays(
                    xy, subsample_to[i], subsample_seed + i
                )
            features.append(xy[0])
            targets.append(xy[1])
            if group_by_class:
                group_ids.append(10 * i + xy[1])
            else:
                group_ids.append(i * torch.ones_like(xy[1]))
        features = torch.cat(features, dim=0)
        targets = torch.cat(targets, dim=0)
        group_ids = torch.cat(group_ids, dim=0)

        inner_dataset = TensorDataset(features, targets)

        self.num_classes = 10

        super().__init__(inner_dataset, group_ids, group_names)


class ImageNetFeatures(GroupedDataset):
    """Dataset object with group structure for ImageNet experiment"""
    def __init__(self, root='data',
                 features_name='resnet50',
                 train=True,
                 subsample_to=-1,
                 subsample_seed=123):
        """

        Parameters
        ----------

        root : string
            Path to the root of the data folder
        features_names : string
            Features expected to be in {root}/imagenet/{features_names}
        train : bool
            If True, loads the train set
        subsample_to : int
            Number of examples to sample
        subsample_seed:
            Seed that decides the randomness of the subsampling.

        """
        split = 'train' if train else 'val'

        # load the base ImageNet object just for reading the class names
        imagenet_base = ImageNet(os.path.join(root, 'imagenet'),
                                 split='val', download=False)
        group_names = ['%03d_' % i + e[0].replace(' ', '_')
                       for i, e in enumerate(imagenet_base.classes)]

        features_path = os.path.join(root, 'imagenet',
                                     features_name + '_features',
                                     split + '.pt')
        data = torch.load(features_path)
        xy = data['features'], data['targets']
        if subsample_to > 0:
            xy = utils.subsample_arrays(
                xy, subsample_to, subsample_seed)

        group_ids = xy[1]

        inner_dataset = TensorDataset(*xy)

        self.num_classes = 1000

        super().__init__(inner_dataset, group_ids, group_names)


class RandomBatchSizeSampler(Sampler):
    """Sampler for batch of data of random size for the multi-level"""

    def __init__(self, dataset,
                 batch_size_min,
                 batch_size_max,
                 doubling_probability=0.5,
                 replace=True,
                 num_batches=None):
        """

        Parameters
        ----------

        dataset : Dataset
        batch_size_min : int
             Minimum of the random batch size
        batch_size_max : int
             Maximum of the random batch size
        doubling_probability : float
             Parameter of the (truncated) geometric law from which we sample
             the batch size
        replace : bool
             If True sample with replacement
        num_batches : int
            Number of batches after which to stop

        """
        self.n = len(dataset)
        self.num_batches = num_batches if num_batches is not None else np.inf
        self.replace = replace
        self.batch_size_min = batch_size_min
        self.batch_size_max = (batch_size_max if batch_size_max > 0
                               else len(dataset))
        self.doubling_probability = doubling_probability

        super().__init__(None)

    def __iter__(self):
        batch_counter = 0
        while True:
            inds = np.random.choice(self.n, replace=self.replace, size=self.n)
            i = 0
            while i < self.n and batch_counter < self.num_batches:
                batch_size = self.sample_batch_size()
                if i + batch_size <= len(inds):
                    yield inds[i:(i + batch_size)]
                else:
                    # returning this instead of breaking and restarting the loop
                    # to avoid selection bias against larger batch sizes
                    yield np.random.choice(self.n, replace=self.replace,
                                           size=batch_size)
                batch_counter += 1
                i += batch_size

            if self.num_batches == np.inf or batch_counter == self.num_batches:
                break

    def __len__(self):
        if self.num_batches != np.inf:
            return self.num_batches
        else:
            return int(self.n / self.batch_size_min * self.doubling_probability)

    def sample_batch_size(self):
        multiplier = 2 ** (np.random.geometric(self.doubling_probability) - 1)
        return np.minimum(self.batch_size_max,
                          self.batch_size_min * multiplier)

    def batch_size_pmf(self, bs):
        factor = np.ceil(np.log2(bs / self.batch_size_min))
        if bs < self.batch_size_max:
            p = ((1 - self.doubling_probability) ** factor
                 * self.doubling_probability)
        else:
            p = (1 - self.doubling_probability) ** factor
        return p


class CustomDistributionSampler(Sampler):
    """Balanced sampling from the labeled and unlabeled data"""

    def __init__(self, dataset,
                 batch_size,
                 replace=True,
                 num_batches=None):
        self.n = len(dataset)
        self.num_batches = (num_batches if num_batches is not None
                            else self.n // batch_size)
        self.batch_size = batch_size
        self.replace = replace
        self.p = np.ones(self.n) / self.n

        super().__init__(None)

    def __iter__(self):
        for _ in range(self.num_batches):
            # pdb.set_trace()
            p = np.maximum(self.p, 0.0)
            p /= p.sum()  # to make absolutely sure that p is in the simplex
                          # otherwise there seem to be random crashes coming
                          # from np.random.choice
            self.inds = np.random.choice(self.n, replace=self.replace,
                                         size=self.batch_size, p=p)
            yield self.inds

    def __len__(self):
        return self.num_batches

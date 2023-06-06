import logging
from abc import abstractmethod
from dataclasses import dataclass
from typing import Sequence, Mapping, Any, List, Optional, Tuple, Union

import pandas as pd
from sklearn.model_selection import train_test_split

import numpy as np


def concat_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Helper function to concatenate values over a set of columns.

    This is useful, for example, as a preprocessing step for performing
    stratified sampling over labels + sensitive attributes."""
    return data.agg(lambda x: ''.join(x.values.astype(str)), axis=1).T


def idx_where_in(x: pd.Series, vals: Sequence[Any]) -> np.ndarray:
    """Return a vector of the numeric indices i where X[i] in vals.

    Note that this function does not differentiate between different numeric
    types; i.e. if [1] (and integer) is in vals and x_j is 1.0 (a float),
     the jth index will be included in the output.
    """
    assert isinstance(vals, list) or isinstance(vals, tuple)
    idxs_bool = x.isin(vals)
    return np.nonzero(idxs_bool.values)[0]


def idx_where_not_in(x: pd.Series, vals: Sequence[Any]) -> np.ndarray:
    """Return a vector of the numeric indices i where X[i] not in vals.

    See note in idx_where_in regarding numeric types.
    """
    assert isinstance(vals, list) or isinstance(vals, tuple)
    idxs_bool = ~x.isin(vals)
    return np.nonzero(idxs_bool.values)[0]


@dataclass
class Splitter:
    """Splitter for non-domain splits."""
    val_size: float
    random_state: int

    @abstractmethod
    def __call__(self, data: pd.DataFrame, labels: pd.Series,
                 groups: pd.DataFrame = None, *args, **kwargs) -> Mapping[
        str, List[int]]:
        """Split a dataset.

        Returns a dictionary mapping split names to indices of the data points
        in that split."""
        raise


class FixedSplitter(Splitter):
    """A splitter for using fixed splits.

    This occurs, for example, when a dataset has a fixed train-test
    split (such as the Adult dataset).

    The FixedSplitter assumes there is a column in the dataset, "Split",
    which contains the values "train", "test".

    Note that for the fixed splitter, val_size indicates what fraction
    **of the training data** should be used for the validation set
    (since we cannot control the fraction of the overall data dedicated
    to validation, due to the prespecified train/test split).
    """

    def __init__(self, split_colname: str="Split", **kwargs):
        self.split_colname = split_colname
        super().__init__(**kwargs)

    def __call__(self, data: pd.DataFrame, labels: pd.Series,
                 groups: pd.DataFrame = None, *args, **kwargs) -> Mapping[
        str, List[int]]:

        assert self.split_colname in data.columns, "data is missing 'Split' column."
        assert all(np.isin(data[self.split_colname], ["train", "test"]))

        test_idxs = np.nonzero((data[self.split_colname] == "test").values)[0]
        train_val_idxs = \
            np.nonzero((data[self.split_colname] == "train").values)[0]

        train_idxs, val_idxs = train_test_split(
            train_val_idxs,
            train_size=(1 - self.val_size),
            random_state=self.random_state)

        del train_val_idxs
        return {"train": train_idxs, "validation": val_idxs, "test": test_idxs}


def _check_input_indices(data: pd.DataFrame):
    """Helper function to validate input indices.

    If a DataFrame is not indexed from (0, n), which happens e.g. when the
    DataFrame has been filtered without resetting the index, it can cause
    major downstream issues with splitting. This is because splitting will
    assume that all values (0,...n) are represented in the index.
    """
    idxs = np.array(sorted(data.index.tolist()))
    expected = np.arange(len(data))
    assert np.all(idxs == expected), "DataFrame is indexed non-sequentially;" \
                                     "try passing the dataframe after "
    return


@dataclass
class RandomSplitter(Splitter):
    test_size: float

    @property
    def train_size(self):
        return 1. - (self.val_size + self.test_size)

    def __call__(self, data: pd.DataFrame, labels: pd.Series,
                 groups: pd.DataFrame = None, *args, **kwargs
                 ) -> Mapping[str, List[int]]:
        _check_input_indices(data)

        idxs = data.index.tolist()
        train_val_idxs, test_idxs = train_test_split(
            idxs,
            test_size=self.test_size,
            random_state=self.random_state)
        train_idxs, val_idxs = train_test_split(
            train_val_idxs,
            train_size=self.train_size / (self.train_size + self.val_size),
            random_state=self.random_state)
        del train_val_idxs
        return {"train": train_idxs, "validation": val_idxs, "test": test_idxs}


@dataclass
class DomainSplitter(Splitter):
    """Splitter for domain splits.

    All observations with domain_split_varname values in domain_split_ood_values
    are placed in the target (test) set; the remaining observations are split
    between the train, validation, and eval set.
    """
    id_test_size: float  # The in-domain test set.
    domain_split_varname: str

    domain_split_ood_values: Optional[Sequence[Any]] = None
    domain_split_id_values: Optional[Sequence[Any]] = None

    # If domain column is greater than this value, observation will be OOD.
    # If less than or equal to this value, observation will be ID.
    domain_split_gt_thresh: Optional[Union[int, float]] = None

    drop_domain_split_col: bool = True  # If True, drop column after splitting.
    ood_val_size: float = 0  # Fraction of OOD data to use for OOD validation set.

    def _split_from_explicit_values(self, domain_vals: pd.Series
                                    ) -> Tuple[np.ndarray, np.ndarray]:

        # Check that either in- or out-of-domain values are specified.
        assert self.is_explicit_split()

        # Check that threshold is not specified, since we are using the
        # explicit list of values to specify ID/OOD.
        assert self.domain_split_gt_thresh is None

        assert isinstance(self.domain_split_ood_values, tuple) \
               or isinstance(self.domain_split_ood_values, list), \
            "domain_split_ood_values must be an iterable type; got type {}".format(
                type(self.domain_split_ood_values))

        ood_vals = self.domain_split_ood_values

        # Fetch the out-of-domain indices.
        ood_idxs = idx_where_in(domain_vals, ood_vals)

        # Fetch the in-domain indices; these are either the explicitly-specified
        # in-domain values, or any values not in the OOD values.

        if self.domain_split_id_values is not None:
            # Check that there is no overlap between train/test domains.
            assert not set(self.domain_split_id_values).intersection(
                set(ood_vals))

            id_idxs = idx_where_in(domain_vals, self.domain_split_id_values)
            if not len(id_idxs):
                raise ValueError(
                    f"No ID observations with {self.domain_split_varname} "
                    f"values {self.domain_split_id_values}; are the values of "
                    f"same type as the column type of {domain_vals.dtype}?")
        else:
            id_idxs = idx_where_not_in(domain_vals, ood_vals)
            if not len(id_idxs):
                raise ValueError(
                    f"No ID observations with {self.domain_split_varname} "
                    f"values not in {ood_vals}.")

        if not len(ood_idxs):
            vals = domain_vals.unique()
            raise ValueError(
                f"No OOD observations with {self.domain_split_varname} values "
                f"{ood_vals}; are the values of same type"
                f" as the column type of {domain_vals.dtype}? Examples of "
                f"values in {self.domain_split_varname}: {vals[:10]}")

        return id_idxs, ood_idxs

    def _split_from_threshold(self, domain_vals: pd.Series) -> Tuple[
        np.ndarray, np.ndarray]:
        """Apply a threshold.

        Values are OOD if > self.domain_split_gt_thresh, else ID."""
        assert self.is_threshold_split()
        assert not self.is_explicit_split()

        if np.any(np.isnan(domain_vals)):
            logging.warning(
                f"detected missing values in domain column prior"
                "to splitting; this can result in unexpected behavior"
                "for threshold-based splits. Any nan values will"
                f"have OOD value: {np.nan > self.domain_split_gt_thresh}")

        ood_idxs = \
            np.nonzero((domain_vals > self.domain_split_gt_thresh).values)[0]
        id_idxs = \
            np.nonzero((domain_vals <= self.domain_split_gt_thresh).values)[0]
        return id_idxs, ood_idxs

    def is_explicit_split(self) -> bool:
        """Helper function to check whether an explicit split is used."""
        return (self.domain_split_ood_values is not None) or (
                self.domain_split_id_values is not None)

    def is_threshold_split(self) -> bool:
        """Helper function to check whether a threshold-based split is used."""
        return (self.domain_split_gt_thresh is not None)

    def __call__(self, data: pd.DataFrame, labels: pd.Series,
                 groups: pd.DataFrame = None, *args, **kwargs) -> Mapping[
        str, List[int]]:
        assert "domain_labels" in kwargs, "domain labels are required."
        domain_vals = kwargs.pop("domain_labels")
        assert isinstance(domain_vals, pd.Series)

        if self.is_explicit_split():
            id_idxs, ood_idxs = self._split_from_explicit_values(domain_vals)

        elif self.is_threshold_split():
            id_idxs, ood_idxs = self._split_from_threshold(domain_vals)

        else:
            raise NotImplementedError("Invalid domain split specified.")

        assert not set(id_idxs).intersection(ood_idxs), "sanity check for " \
                                                        "nonoverlapping " \
                                                        "domain split"
        assert not set(domain_vals.iloc[id_idxs]) \
            .intersection(domain_vals.iloc[ood_idxs]), "sanity check for no " \
                                                       "domain leakage"

        train_idxs, id_valid_eval_idxs = train_test_split(
            id_idxs, test_size=(self.val_size + self.id_test_size),
            random_state=self.random_state)

        valid_idxs, id_test_idxs = train_test_split(
            id_valid_eval_idxs,
            test_size=self.id_test_size / (self.val_size + self.id_test_size),
            random_state=self.random_state)

        outputs = {"train": train_idxs, "validation": valid_idxs,
                   "id_test": id_test_idxs}

        # Out-of-distribution splits
        if self.ood_val_size:
            ood_test_idxs, ood_valid_idxs = train_test_split(
                ood_idxs,
                test_size=self.ood_val_size,
                random_state=self.random_state)
            outputs["ood_test"] = ood_test_idxs
            outputs["ood_validation"] = ood_valid_idxs

        else:
            outputs["ood_test"] = ood_idxs

        return outputs

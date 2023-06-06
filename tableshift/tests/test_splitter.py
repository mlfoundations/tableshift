"""
Tests for splitters.

To run tests: python -m unittest tableshift/tests/test_splitter.py -v
"""
import unittest
import pandas as pd
import numpy as np

from tableshift.core.splitter import DomainSplitter

np.random.seed(54329)


class TestDomainSplitterSplits(unittest.TestCase):

    def setUp(self) -> None:
        n = 5000
        self.data = pd.DataFrame({
            "values1": np.arange(n),
            "values2": np.random.choice([1, 2, 3], n),
            "domain": (["a"] * int(n / 2) + ["b"] * int(n / 2))
        })

        self.groups = pd.DataFrame({
            "group_var_a": np.random.choice([0, 1], n),
            "group_var_b": np.random.choice([0, 1], n),
        })

        self.labels = pd.Series(np.random.choice([0, 1], n))

    def test_disjoint_splits(self):
        data = self.data
        groups = self.groups
        labels = self.labels
        ood_vals = ["a"]
        splitter = DomainSplitter(id_test_size=0.5,
                                  val_size=0.1,
                                  domain_split_varname="domain",
                                  domain_split_ood_values=ood_vals,
                                  ood_val_size=0.25,
                                  random_state=45378)

        splits = splitter(data, labels, groups=groups,
                          domain_labels=data["domain"])

        train_domains = data.iloc[splits["train"]]["domain"]
        val_domains = data.iloc[splits["validation"]]["domain"]
        ood_val_domains = data.iloc[splits["ood_validation"]]["domain"]
        id_test_domains = data.iloc[splits["id_test"]]["domain"]
        ood_test_domains = data.iloc[splits["ood_test"]]["domain"]

        # Check that OOD splits only contain OOD values
        self.assertTrue(np.all(ood_test_domains.isin(ood_vals)))
        self.assertTrue(np.all(ood_val_domains.isin(ood_vals)))

        # Check that ID splits do not contain any OOD values
        self.assertFalse(np.any(train_domains.isin(ood_vals)))
        self.assertFalse(np.any(val_domains.isin(ood_vals)))
        self.assertFalse(np.any(id_test_domains.isin(ood_vals)))

        # Check for proper partitioning of id/ood
        assert set(train_domains) == set(id_test_domains)
        self.assertTrue(set(train_domains).isdisjoint(set(ood_test_domains)))
        self.assertTrue(set(val_domains).isdisjoint(set(ood_val_domains)))
        self.assertTrue(set(id_test_domains).isdisjoint(set(ood_test_domains)))

        # Check that output size is same as input
        self.assertEqual(sum(len(x) for x in splits.values()), len(data))

        # Check that every index is somewhere in splits
        all_idxs = set(idx for split_idxs in splits.values()
                       for idx in split_idxs)
        self.assertEqual(all_idxs, set(data.index.tolist()))

    def test_no_grouper(self):
        data = self.data
        labels = self.labels
        ood_vals = ["a"]
        splitter = DomainSplitter(id_test_size=0.5,
                                  val_size=0.1,
                                  domain_split_varname="domain",
                                  domain_split_ood_values=ood_vals,
                                  ood_val_size=0.25,
                                  random_state=45378)

        splits = splitter(data, labels, groups=None,
                          domain_labels=data["domain"])

        # Check that output size is same as input
        self.assertEqual(sum(len(x) for x in splits.values()), len(data))

        # Check that every index is somewhere in splits
        all_idxs = set(idx for split_idxs in splits.values()
                       for idx in split_idxs)
        self.assertEqual(all_idxs, set(data.index.tolist()))


class TestDomainSplitterDtypes(unittest.TestCase):
    """Test DomainSplitter with different types of ood values."""

    def test_float_vs_int(self):
        """Test case where domain values are floats and OOD values are ints."""
        n = 5000
        ood_vals = [2]
        data = pd.DataFrame({
            "values1": np.arange(n),
            "values2": np.random.choice([1, 2, 3], n),
            "domain": ([1.0] * int(n / 2) + [2.0] * int(n / 2))
        })

        labels = pd.Series(np.random.choice([0, 1], n))

        splitter = DomainSplitter(id_test_size=0.5,
                                  val_size=0.1,
                                  domain_split_varname="domain",
                                  domain_split_ood_values=ood_vals,
                                  ood_val_size=0.25,
                                  random_state=45478)

        splits = splitter(data, labels, domain_labels=data["domain"])

        # Check that number of OOD observations over all splits matches
        # number of OOD observations in the original data.
        ood_elems_in_splits = sum(
            len(v) for k, v in splits.items() if "ood" in k)
        ood_elems_in_data = np.isin(data["domain"].values, ood_vals).sum()
        self.assertEqual(ood_elems_in_data, ood_elems_in_splits)

    def test_int_vs_float(self):
        """Test case where domain values are ints and OOD values are floats."""
        n = 5000
        ood_vals = [2.0]
        data = pd.DataFrame({
            "values1": np.arange(n),
            "values2": np.random.choice([1, 2, 3], n),
            "domain": ([1] * int(n / 2) + [2] * int(n / 2))
        })

        labels = pd.Series(np.random.choice([0, 1], n))

        splitter = DomainSplitter(id_test_size=0.5,
                                  val_size=0.1,
                                  domain_split_varname="domain",
                                  domain_split_ood_values=ood_vals,
                                  ood_val_size=0.25,
                                  random_state=41378)

        splits = splitter(data, labels, domain_labels=data["domain"])

        # Check that number of OOD observations over all splits matches
        # number of OOD observations in the original data.
        ood_elems_in_splits = sum(
            len(v) for k, v in splits.items() if "ood" in k)
        ood_elems_in_data = np.isin(data["domain"].values, ood_vals).sum()
        self.assertEqual(ood_elems_in_data, ood_elems_in_splits)


class TestThresholdDomainSplitter(unittest.TestCase):
    def test_float_split(self):

        data = pd.DataFrame(
            {"values1": np.arange(10),
             "values2": np.arange(10),
             "domain": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}
        )
        labels = pd.Series(np.random.choice([0, 1], 10))

        thresh = 0.55
        splitter = DomainSplitter(id_test_size=0.2, val_size=0.2,
                                  ood_val_size=0.2,
                                  random_state=43890,
                                  domain_split_varname="domain",
                                  domain_split_gt_thresh=thresh)
        splits = splitter(data, labels, domain_labels=data["domain"])

        # Check splits have expected size
        self.assertEqual(len(splits["train"]), 3)
        self.assertEqual(len(splits["id_test"]), 1)
        self.assertEqual(len(splits["ood_test"]), 4)
        self.assertEqual(len(splits["validation"]), 1)
        self.assertEqual(len(splits["ood_validation"]), 1)

        # Check splits have expected domain values
        for split in ("train", "validation", "id_test"):
            vals = data.loc[splits[split]]["domain"]
            self.assertTrue(np.all(vals <= thresh))

        for split in ("ood_validation", "ood_test"):
            vals = data.loc[splits[split]]["domain"]
            self.assertTrue(np.all(vals > thresh))

    def test_int_split(self):
        # half ID, half OOD
        domain_vals = ([10] * 50) + ([20] * 50)

        data = pd.DataFrame(
            {"values1": np.arange(100),
             "values2": np.arange(100),
             "domain": domain_vals}
        )
        labels = pd.Series(np.random.choice([0, 1], 100))

        thresh = 15
        splitter = DomainSplitter(id_test_size=0.1, val_size=0.1,
                                  ood_val_size=0.2,
                                  random_state=463890,
                                  domain_split_varname="domain",
                                  domain_split_gt_thresh=thresh)
        splits = splitter(data, labels, domain_labels=data["domain"])

        # Check splits have expected size
        self.assertEqual(len(splits["train"]), 40)
        self.assertEqual(len(splits["id_test"]), 5)
        self.assertEqual(len(splits["validation"]), 5)
        self.assertEqual(len(splits["ood_validation"]), 10)
        self.assertEqual(len(splits["ood_test"]), 40)

        # Check splits have expected domain values
        for split in ("train", "validation", "id_test"):
            vals = data.loc[splits[split]]["domain"]
            self.assertTrue(np.all(vals <= thresh))

        for split in ("ood_validation", "ood_test"):
            vals = data.loc[splits[split]]["domain"]
            self.assertTrue(np.all(vals > thresh))

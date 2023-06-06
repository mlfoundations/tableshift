"""
Tests for TabularDatasets.

To run tests: python -m unittest tableshift/tests/test_tabular_dataset.py -v
"""
import logging
import tempfile
import unittest

from tableshift.core import TabularDataset, RandomSplitter, \
    DatasetConfig, PreprocessorConfig, CachedDataset

LOG_LEVEL = logging.DEBUG

logger = logging.getLogger()
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    level=LOG_LEVEL,
    datefmt='%Y-%m-%d %H:%M:%S')


class TestTabularDataset(unittest.TestCase):
    def test_init_dataset_german(self):
        """
        Initialize the German dataset. Checks that an exception is not raised.
        """
        with tempfile.TemporaryDirectory() as td:
            _ = TabularDataset(name="german",
                               config=DatasetConfig(cache_dir=td),
                               grouper=None,
                               splitter=RandomSplitter(val_size=0.25,
                                                       random_state=68594,
                                                       test_size=0.25),
                               preprocessor_config=PreprocessorConfig(
                                   passthrough_columns="all"))
        return

    def test_get_pandas_adult(self):
        """
        Initialize the Adult dataset and check for nonempty train/test/val DFs.
        """
        with tempfile.TemporaryDirectory() as td:
            dset = TabularDataset(name="adult",
                                  config=DatasetConfig(cache_dir=td),
                                  grouper=None,
                                  splitter=RandomSplitter(val_size=0.25,
                                                          random_state=68594,
                                                          test_size=0.25),
                                  preprocessor_config=PreprocessorConfig(
                                      passthrough_columns="all"))
            for split in ("train", "validation", "test"):
                X, y, _, _ = dset.get_pandas(split)
                self.assertTrue(len(X) != 0)
                self.assertTrue(len(X) == len(y))
        return


class TestCachedDataset(unittest.TestCase):
    def _test_cache_and_load(self, name: str):
        """Cache a dataset and reload it."""
        with tempfile.TemporaryDirectory() as td:
            preprocessor_config = PreprocessorConfig(passthrough_columns="all")
            dataset_config = DatasetConfig(cache_dir=td)
            dset = TabularDataset(name=name,
                                  config=dataset_config,
                                  grouper=None,
                                  splitter=RandomSplitter(val_size=0.25,
                                                          random_state=68594,
                                                          test_size=0.25),
                                  preprocessor_config=preprocessor_config)
            dset.to_sharded(rows_per_shard=64)

            cached_dset = CachedDataset(config=dataset_config,
                                        name=name,
                                        initialize_data=True,
                                        preprocessor_config=preprocessor_config)
            for split in ("train", "validation", "test"):
                dset_split = dset.get_pandas(split)
                cached_dset_split = cached_dset.get_pandas(split)
                self.assertTrue(
                    dset_split[0].shape == cached_dset_split[0].shape,
                    msg="X shapes not identical for split %s" % split)
                self.assertListEqual(
                    dset_split[0].columns.tolist(),
                    cached_dset_split[0].columns.tolist(),
                    msg='Feature names not identical for split %s' % split)
                self.assertTrue(
                    dset_split[1].shape == cached_dset_split[1].shape,
                    msg="y shapes not identical for split %s" % split)

    def test_cache_and_load_german(self):
        """Cache the German dataset and reload it."""
        self._test_cache_and_load("german")

    def test_cache_and_load_adult(self):
        """Cache the Adult dataset and reload it."""
        self._test_cache_and_load("adult")

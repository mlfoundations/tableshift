"""
Tests for TabularDatasets.

To run tests: python -m unittest tableshift/tests/test_getters.py -v
"""
import unittest

from tableshift import get_dataset, get_iid_dataset


class TestGetters(unittest.TestCase):
    """Test getters for some (small, public) datasets."""

    def test_get_dataset_german(self):
        _ = get_dataset("german")

    def test_get_iid_dataset_german(self):
        _ = get_iid_dataset("german")

    def test_get_dataset_adult(self):
        _ = get_dataset("adult")

    def test_get_iid_dataset_adult(self):
        _ = get_iid_dataset("adult")

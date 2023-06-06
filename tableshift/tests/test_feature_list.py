"""
Tests for FeatureLists.

To run tests: python -m unittest tableshift/tests/test_feature_list.py -v
"""
import unittest

from tableshift.core.features import Feature, FeatureList, cat_dtype


class TestFeatureList(unittest.TestCase):
    def test_add_features(self):
        fl1 = FeatureList(features=[
            Feature("intfeat1", int),
            Feature("floatfeat1", float),
            Feature("catfeat1", cat_dtype)])
        fl2 = FeatureList(features=[
            Feature("intfeat2", int),
            Feature("floatfeat2", float),
            Feature("catfeat2", cat_dtype)])
        new_fl = fl1 + fl2

        # Check output feature list
        self.assertListEqual(
            sorted(new_fl.names),
            sorted(["intfeat1", "floatfeat1", "catfeat1",
                    "intfeat2", "floatfeat2", "catfeat2"]))

        # Check that original FeatureList objects are not modified.
        self.assertListEqual(sorted(fl1.names),
                             sorted(["intfeat1", "floatfeat1", "catfeat1"]))
        self.assertListEqual(sorted(fl2.names),
                             sorted(["intfeat2", "floatfeat2", "catfeat2"]))

    def test_add_multi_target(self):
        fl1 = FeatureList(features=[
            Feature("intfeat1", int),
            Feature("floatfeat1", float, is_target=True)])
        fl2 = FeatureList(features=[
            Feature("intfeat2", int),
            Feature("floatfeat2", float, is_target=True)])
        with self.assertRaises(ValueError):
            fl1 + fl2

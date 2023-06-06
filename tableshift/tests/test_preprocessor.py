"""
Tests for Preprocessor objects.

To run tests: python -m unittest tableshift/tests/test_preprocessor.py -v

To run a specific individual test:
python -m unittest tableshift/tests/test_preprocessor.py \
    tableshift.tests.test_preprocessor.TestPreprocessor.test_map_name -v
"""

import copy
import unittest
import numpy as np
import pandas as pd
from tableshift.core.features import Preprocessor, PreprocessorConfig, \
    FeatureList, Feature, cat_dtype, remove_verbose_prefixes


class TestPreprocessor(unittest.TestCase):
    def setUp(self) -> None:
        n = 100
        self.df = pd.DataFrame({
            "int_a": np.arange(0, n, dtype=int),
            "int_b": np.arange(-n, 0, dtype=int),
            "float_a": np.random.uniform(size=n),
            "float_b": np.random.uniform(-1., 1., size=n),
            "string_a": np.random.choice(["a", "b", "c"], size=n),
            "cat_a": pd.Categorical(
                np.random.choice(["typea", "typeb"], size=n)),
        })
        return

    def test_passthrough_target(self):
        """Test that the target feature is not transformed.

        This tests mimics/tests the pattern used in tabular_dataset.py."""
        feature_list = FeatureList([
            Feature("int_a", int, is_target=True,
                    value_mapping={x: -x for x in range(len(self.df))}),
            Feature("int_b", int),
            Feature("float_a", float),
            Feature("float_b", float),
            Feature("string_a", str, value_mapping={
                "a": "Diagnosis of disease A",
                "b": "Diagnosis of disease B",
                "c": "Diagnosis of disease C"}),
            Feature("cat_a", cat_dtype),
        ])
        for num_feat_handling in ("map_values", "normalize", "kbins"):
            preprocessor = Preprocessor(
                config=PreprocessorConfig(numeric_features=num_feat_handling),
                feature_list=feature_list)
            data = copy.deepcopy(self.df)
            train_idxs = list(range(50))
            transformed = preprocessor.fit_transform(
                data, train_idxs=train_idxs,
                passthrough_columns=[feature_list.target])
            self.assertListEqual(
                self.df[feature_list.target].tolist(),
                transformed[feature_list.target].tolist(),
                msg=f'Target was transformed with handling {num_feat_handling}')

    def test_passthrough_all(self):
        """Test case with no transformations (passthrough="all")."""
        data = copy.deepcopy(self.df)
        preprocessor = Preprocessor(
            config=PreprocessorConfig(passthrough_columns="all"))
        train_idxs = list(range(50))
        transformed = preprocessor.fit_transform(data, train_idxs=train_idxs)

        # Check that values are unmodified
        np.testing.assert_array_equal(data.values, transformed.values)
        # Check that dtypes are the same
        self.assertListEqual(data.dtypes.tolist(),
                             transformed.dtypes.tolist())
        # Check that feature names are the same
        self.assertListEqual(sorted(transformed.columns.tolist()),
                             sorted(self.df.columns.tolist()))
        return

    def test_passthrough_numeric(self):
        """Test case with no numeric transformations."""
        data = copy.deepcopy(self.df)
        preprocessor = Preprocessor(
            config=PreprocessorConfig(numeric_features="passthrough"))
        train_idxs = list(range(50))
        transformed = preprocessor.fit_transform(data, train_idxs=train_idxs)
        numeric_cols = ["int_a", "int_b", "float_a", "float_b"]

        # Check that values of numeric cols are unmodified
        np.testing.assert_array_equal(data[numeric_cols].values,
                                      transformed[numeric_cols].values)
        # Check that dtypes of numeric cols are the same
        self.assertListEqual(data.dtypes[numeric_cols].tolist(),
                             transformed.dtypes[numeric_cols].tolist())
        return

    def test_passthrough_categorical(self):
        """Test case with no categorical transformations."""
        data = copy.deepcopy(self.df)
        preprocessor = Preprocessor(
            config=PreprocessorConfig(categorical_features="passthrough"))
        train_idxs = list(range(50))
        transformed = preprocessor.fit_transform(data, train_idxs=train_idxs)
        categorical_cols = ["string_a", "cat_a"]

        # Check that values of categorical cols are unmodified
        np.testing.assert_array_equal(data[categorical_cols].values,
                                      transformed[categorical_cols].values)
        # Check that dtypes of numeric cols are the same
        self.assertListEqual(data.dtypes[categorical_cols].tolist(),
                             transformed.dtypes[categorical_cols].tolist())
        return

    def test_label_encoder(self):
        preprocessor_config = PreprocessorConfig(
            categorical_features="label_encode")
        preprocessor = Preprocessor(config=preprocessor_config)
        data = copy.deepcopy(self.df)
        train_idxs = list(range(50))
        transformed = preprocessor.fit_transform(data=data,
                                                 train_idxs=train_idxs)
        for feat in ("string_a", "cat_a"):
            self.assertTrue(
                np.issubdtype(transformed[feat].dtype, float) or
                np.issubdtype(transformed[feat].dtype, int))

    def test_map_values(self):
        """Test mapping of values."""

        feature_list = FeatureList([
            Feature("int_a", int,
                    value_mapping={x: -x for x in range(len(self.df))}),
            Feature("int_b", int),
            Feature("float_a", float),
            Feature("float_b", float),
            Feature("string_a", str, value_mapping={
                "a": "Diagnosis of disease A",
                "b": "Diagnosis of disease B",
                "c": "Diagnosis of disease C"}),
            Feature("cat_a", cat_dtype),
        ])
        data = copy.deepcopy(self.df)
        preprocessor = Preprocessor(
            config=PreprocessorConfig(
                categorical_features="map_values",
                numeric_features="map_values"),
            feature_list=feature_list)
        train_idxs = list(range(50))
        transformed = preprocessor.fit_transform(data, train_idxs=train_idxs)

        for f in feature_list.features:
            if f.value_mapping is not None:
                fname = f.name
                self.assertTrue(fname in transformed.columns)

                self.assertTrue(np.all(
                    np.isin(transformed[fname].values,
                            np.array(list(f.value_mapping.values())))
                ),
                    msg=f"failed for feature {fname}")

    def test_map_values_all(self):
        """Test case where a value map is used for *all* features."""

        feature_list = FeatureList([
            Feature("int_a", int,
                    value_mapping={x: -x for x in range(len(self.df))}),
            Feature("int_b", int,
                    value_mapping={x: x % 2 for x in
                                   self.df["int_b"].unique()}),
            Feature("float_a", float,
                    value_mapping={x: x + 1000 for x in
                                   self.df["float_a"].unique()}
                    ),
            Feature("float_b", float,
                    value_mapping={x: x > 0 for x in
                                   self.df["float_b"].unique()}),
            Feature("string_a", str, value_mapping={
                "a": "Diagnosis of disease A",
                "b": "Diagnosis of disease B",
                "c": "Diagnosis of disease C"}),
            Feature("cat_a", cat_dtype,
                    value_mapping={"typea": "Patient of Type A",
                                   "typeb": "Patient of Type B"}),
        ])
        data = copy.deepcopy(self.df)
        preprocessor = Preprocessor(
            config=PreprocessorConfig(
                categorical_features="map_values",
                numeric_features="map_values"),
            feature_list=feature_list)
        train_idxs = list(range(50))
        transformed = preprocessor.fit_transform(data, train_idxs=train_idxs)
        for f in feature_list.features:
            fname = f.name
            self.assertTrue(fname in transformed.columns)

            self.assertTrue(np.all(
                np.isin(transformed[fname].values,
                        np.array(list(f.value_mapping.values())))
            ),
                msg=f"failed for feature {fname}")

    def test_map_name(self):
        """Test mapping of extended feature names."""

        feature_list = FeatureList([
            Feature("int_a", int,
                    name_extended="Integer A value"),
            Feature("int_b", int,
                    name_extended="Integer B value",
                    is_target=True),
            Feature("float_a", float),
            Feature("float_b", float),
            Feature("string_a", str,
                    name_extended="String A value"),
            Feature("cat_a", cat_dtype),
        ])
        data = copy.deepcopy(self.df)
        preprocessor = Preprocessor(
            config=PreprocessorConfig(use_extended_names=True,
                                      numeric_features="passthrough",
                                      categorical_features="passthrough"),
            feature_list=feature_list)
        train_idxs = list(range(50))

        transformed = preprocessor.fit_transform(data, train_idxs=train_idxs)

        expected_names = ["Integer A value", "int_b", "float_a", "float_b",
                          "String A value", "cat_a"]

        # Check that names are mapped
        self.assertListEqual(transformed.columns.tolist(), expected_names)
        # Check that data is unchanged
        np.testing.assert_array_equal(transformed.values, self.df.values)

    def test_remove_verbose_prefixes(self):
        colnames = ["onehot__x_feature", "scale__float0",
                    "kbin__myfeature", "map__cat_feature"]
        expected = ["x_feature", "float0",
                    "myfeature", "cat_feature"]
        output = remove_verbose_prefixes(colnames)
        self.assertListEqual(output, expected)

    def test_remove_verbose_prefixes_in_pipeline_onehot_normalize(self):
        """End-to-end test that verbose prefixes are removed, with onehot/norm.
        """
        data = copy.deepcopy(self.df)
        preprocessor = Preprocessor(config=PreprocessorConfig(
            categorical_features="one_hot",
            numeric_features="normalize",
        ))
        train_idxs = list(range(50))

        expected_output_columns = ['int_a', 'int_b', 'float_a', 'float_b',
                                   'string_a_c', 'string_a_b',
                                   'string_a_a', 'cat_a_typeb', 'cat_a_typea']
        transformed = preprocessor.fit_transform(
            data, train_idxs=train_idxs)

        self.assertListEqual(sorted(transformed.columns.tolist()),
                             sorted(expected_output_columns))

    def test_remove_verbose_prefixes_in_pipeline_map_bin(self):
        """End-to-end test that verbose prefixes are removed, with mapping
        and binning.
        """
        data = copy.deepcopy(self.df)

        feature_list = FeatureList([
            Feature("int_a", int,
                    value_mapping={x: -x for x in range(len(self.df))}),
            Feature("int_b", int),
            Feature("float_a", float),
            Feature("float_b", float),
            Feature("string_a", str,
                    value_mapping={
                        "a": "Diagnosis of disease A",
                        "b": "Diagnosis of disease B",
                        "c": "Diagnosis of disease C"}),
            Feature("cat_a", cat_dtype),
        ])

        preprocessor = Preprocessor(config=PreprocessorConfig(
            categorical_features="map_values",
            numeric_features="kbins"),
            feature_list=feature_list)
        train_idxs = list(range(50))

        expected_output_columns = self.df.columns.tolist()
        transformed = preprocessor.fit_transform(
            data, train_idxs=train_idxs)

        self.assertListEqual(sorted(transformed.columns.tolist()),
                             sorted(expected_output_columns))

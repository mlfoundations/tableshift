"""
Utilities for the MIMIC-Extract dataset.

MIMIC-Extract requires access to the MIMIC dataset, which is a public
credentialized dataset. Obtain access from the Physionet website.
See the instructions at the links below.

For more information on datasets and access in TableShift, see:
* https://tableshift.org/datasets.html
* https://github.com/mlfoundations/tableshift
"""

import logging
from typing import List, Union

import pandas as pd

from tableshift.core.features import Feature, FeatureList, cat_dtype
from tableshift.core.utils import sub_illegal_chars
from tableshift.datasets.mimic_extract_feature_lists import \
    MIMIC_EXTRACT_SHARED_FEATURES, _MIMIC_EXTRACT_LOS_3_SELECTED_FEATURES, \
    _MIMIC_EXTRACT_MORT_HOSP_SELECTED_FEATURES
from tableshift.datasets.utils import convert_numeric_dtypes, complete_cases

ID_COLS = ['subject_id', 'hadm_id', 'icustay_id']

MIMIC_EXTRACT_STATIC_FEATURES = FeatureList(features=[
    Feature("gender", cat_dtype, "Gender (M/F)."),
    Feature("age", float, "Age."),
    Feature("ethnicity", cat_dtype, "Ethnicity (41 unique values)."),
    Feature("insurance", cat_dtype,
            "Medicare, Private, Medicaid, Government, Self Pay.")
])

MIMIC_EXTRACT_LOS_3_FEATURES = FeatureList(features=[
    *MIMIC_EXTRACT_STATIC_FEATURES,
    *MIMIC_EXTRACT_SHARED_FEATURES,
    Feature('los_3', int, is_target=True)
])

MIMIC_EXTRACT_MORT_HOSP_FEATURES = FeatureList(features=[
    *MIMIC_EXTRACT_STATIC_FEATURES,
    *MIMIC_EXTRACT_SHARED_FEATURES,
    Feature('mort_hosp', int, is_target=True)
])

MIMIC_EXTRACT_LOS_3_SELECTED_FEATURES = FeatureList(features=[
    *_MIMIC_EXTRACT_LOS_3_SELECTED_FEATURES,
    Feature('los_3', int, is_target=True)
])
MIMIC_EXTRACT_MORT_HOSP_SELECTED_FEATURES = FeatureList(features=[
    *_MIMIC_EXTRACT_MORT_HOSP_SELECTED_FEATURES,
    Feature('mort_hosp', int, is_target=True)
])


def simple_imputer(df: pd.DataFrame) -> pd.DataFrame:
    """Via https://github.com/MLforHealth/MIMIC_Extract/blob/
    master/notebooks/Baselines%20for%20Mortality%20and%20LOS
    %20prediction%20-%20Sklearn.ipynb """
    idx = pd.IndexSlice
    df = df.copy()
    if len(df.columns.names) > 2: df.columns = df.columns.droplevel(
        ('label', 'LEVEL1', 'LEVEL2'))

    df_out = df.loc[:, idx[:, ['mean', 'count']]]
    icustay_means = df_out.loc[:, idx[:, 'mean']].groupby(ID_COLS).mean()

    df_out.loc[:, idx[:, 'mean']] = df_out.loc[:, idx[:, 'mean']].groupby(
        ID_COLS).fillna(
        method='ffill'
    ).groupby(ID_COLS).fillna(icustay_means).fillna(0)

    df_out.loc[:, idx[:, 'count']] = (df.loc[:, idx[:, 'count']] > 0).astype(
        float)
    df_out.rename(columns={'count': 'mask'}, level='Aggregation Function',
                  inplace=True)

    is_absent = (1 - df_out.loc[:, idx[:, 'mask']])
    hours_of_absence = is_absent.cumsum()
    time_since_measured = hours_of_absence - hours_of_absence[
        is_absent == 0].fillna(method='ffill')
    time_since_measured.rename(columns={'mask': 'time_since_measured'},
                               level='Aggregation Function', inplace=True)

    df_out = pd.concat((df_out, time_since_measured), axis=1)
    df_out.loc[:, idx[:, 'time_since_measured']] = \
        df_out.loc[:, idx[:, 'time_since_measured']].fillna(100)

    df_out.sort_index(axis=1, inplace=True)
    return df_out


def preprocess_mimic_extract(df: pd.DataFrame, task: str,
                             static_features: List[str]) -> pd.DataFrame:
    """Apply the default MIMIC-extract preprocessing.

    Specifically, this includes flattening the data and imputing missing
    values. """
    # Remove Ys and static features; this allows us to use the same
    # preprocessing code to flatten the time-varying features as in the MIMIC
    # provided notebooks.

    assert task in df.columns, f"task {task} not in dataframe columns."
    Ys = df.pop(task)
    # Extract statics via multicolumn 'pop'
    statics = df[static_features]
    df.drop(columns=static_features, inplace=True)

    # Merging with "flat" (non-hierarchical) static features and labels
    # flattens the MultiIndex of the data. Here we restore it, so that we can
    # use the MIMIC-extract imputer.
    df.columns = pd.MultiIndex.from_tuples(
        df.columns, names=['LEVEL2', 'Aggregation Function'])

    logging.debug(f"performing imputation on dataframe of shape {df.shape}")
    df = simple_imputer(df)
    logging.debug(f"pivoting dataframe of shape {df.shape}")
    flattened_df = df.pivot_table(index=['subject_id', 'hadm_id', 'icustay_id'],
                                  columns=['hours_in'])

    # Flatten the columns prior to joining and clean up column names
    flattened_df.columns = flattened_df.columns.to_flat_index()
    flattened_df.columns = ["_".join(str(x).replace(' ', '_') for x in c) for c
                            in flattened_df.columns]

    # All of these targets do not vary based on time, so we can drop the
    # 'hours_in' level (which is not present in the flattened DataFrame index
    # either after flattening) and merge on the remaining 3-level index with
    # levels (subject_id, hadm_id, icustay_id).
    NON_TIME_VARYING_TARGETS = ('los_3', 'los_7', 'mort_hosp', 'mort_icu')
    assert Ys.name in NON_TIME_VARYING_TARGETS, \
        f"sanity check that label {Ys.name} is not time-varying."

    def _drop_hours_in_index(df_in: Union[pd.Series, pd.DataFrame]) -> Union[
        pd.Series, pd.DataFrame]:
        """Helper function to remove the hours_in index level."""
        assert df_in.index.names == ['subject_id', 'hadm_id', 'icustay_id',
                                     'hours_in']
        df_in = df_in.droplevel('hours_in')
        return df_in[~df_in.index.duplicated(keep='first')]

    # Drop the hours_in from index and then drop duplicate indices (the
    # values are the same at every value of hours_in for each unique (
    # subject_id, hadm_id, icustay_id) index, since the label and static
    # values are all not time-varying.
    Ys = _drop_hours_in_index(Ys)
    statics = _drop_hours_in_index(statics)

    flattened_df_rows_pre_joins = len(flattened_df)
    flattened_df = flattened_df.join(Ys, how="inner")
    flattened_df = flattened_df.join(statics, how="inner")
    assert len(flattened_df) == flattened_df_rows_pre_joins, \
        "sanity check no data loss when joining data to labels/statics."

    # Remove the index and verify that all of the index levels are unique.
    # This should be the case because MIMIC-extract only keeps the first ICU
    # visit for each patient. This ensures that any downstream splitting does
    # not leak data across subject_id/hadm_id/icustay_id.
    idxnames = flattened_df.index.names
    flattened_df.reset_index(inplace=True)
    for idxname in idxnames:
        assert flattened_df[idxname].nunique() == len(flattened_df), \
            f"values for index level {idxname} are not unique."
    flattened_df.drop(columns=idxnames, inplace=True)

    len_pre_drop = len(flattened_df)
    flattened_df = complete_cases(flattened_df)
    logging.info(
        f"dropped {len_pre_drop - len(flattened_df)} rows "
        f"({100 * (len_pre_drop - len(flattened_df)) / len_pre_drop:.2f})% "
        f"of data) containing missing values "
        f"after imputation (could be due to missing static features).")

    flattened_df = convert_numeric_dtypes(flattened_df)

    # Clean up variable names, since most columns are passed-through by the
    # TableShift preprocessor.
    flattened_df.columns = [sub_illegal_chars(c) for c in flattened_df.columns]

    return flattened_df

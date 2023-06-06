import pandas as pd

from tableshift.core.features import Feature, FeatureList, cat_dtype

COMPAS_RESOURCES = [
    "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"]

COMPAS_FEATURES = FeatureList(features=[
    Feature('juv_fel_count', int,
            name_extended='Count of juvenile felonies'),
    Feature('juv_misd_count', int,
            name_extended='Count of juvenile misemeanors'),
    Feature('juv_other_count', int,
            name_extended='Count of other juvenile records'),
    Feature('priors_count', int,
            name_extended='Count of prior convictions'),
    Feature('age', int),
    Feature('c_charge_degree', cat_dtype, name_extended='Charge degree',
            value_mapping={'F': 'Felony', 'M': 'Misdemeanor'}),
    Feature('sex', cat_dtype),
    Feature('race', cat_dtype),
    Feature('Target', float, is_target=True),
])


def preprocess_compas(df: pd.DataFrame):
    """Preprocess COMPAS dataset.

    See https://github.com/RuntianZ/doro/blob/master/compas.py .
    """

    columns = ['juv_fel_count', 'juv_misd_count', 'juv_other_count',
               'priors_count',
               'age',
               'c_charge_degree',
               'sex', 'race', 'is_recid', 'compas_screening_date']

    df = df[['id'] + columns].drop_duplicates()
    df = df[columns]

    df.rename(columns={"is_recid": "Target"}, inplace=True)
    return df

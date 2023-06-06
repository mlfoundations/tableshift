from dataclasses import dataclass
from typing import Callable, Union

import folktables
import frozendict
import numpy as np
import pandas as pd

from tableshift.core.features import Feature, FeatureList, cat_dtype

################################################################################
# Shared features used in more than one task
################################################################################

NWLK_FEATURE = Feature('NWLK', cat_dtype, "Looking for work",
                       name_extended="Looking for work",
                       value_mapping={'01': 'Yes', '02': 'No',
                                      '03': 'Did not report'})

NWLA_FEATURE = Feature('NWLA', cat_dtype, "On layoff from work",
                       name_extended="On layoff from work",
                       value_mapping={'01': 'Yes', '02': 'No',
                                      '03': 'Did not report'})

FER_FEATURE = Feature('FER', cat_dtype,
                      "Gave birth to child within the past 12 months",
                      name_extended="Gave birth to child within the past 12 months",
                      value_mapping={
                          '00': "less than 15 years old or greater than 50 years or male",
                          '01': "Yes", '02': "No", })

DREM_FEATURE = Feature('DREM', cat_dtype, "Cognitive difficulty",
                       name_extended="Cognitive difficulty",
                       value_mapping={'01': 'Yes', '02': 'No'})

DEYE_FEATURE = Feature('DEYE', cat_dtype, "Vision difficulty",
                       name_extended="Vision difficulty",
                       value_mapping={'01': 'Yes', '02': 'No'})

ANC_FEATURE = Feature('ANC', cat_dtype, "Ancestry recode",
                      name_extended="Ancestry",
                      value_mapping={'1': 'Single', '2': 'Multiple',
                                     '3': 'Unclassified',
                                     '4': 'Not reported',
                                     '8': 'Ancestry suppressed for this group'})

MIL_FEATURE = Feature('MIL', cat_dtype, "Military service",
                      name_extended="Military service",
                      value_mapping={'00': 'N/A (less than 17 years old)',
                                     '01': 'Now on active duty',
                                     '02': 'On active duty in the past, but not now',
                                     '03': 'Only on active duty for training in Reserves/National Guard',
                                     '04': 'Never served in the military'})

ESP_FEATURE = Feature('ESP', cat_dtype, "Employment status of parents",
                      name_extended="Employment status of parents",
                      value_mapping={
                          '00': 'Not own child of householder, and not child in subfamily',
                          '01': 'Living with two parents: both parents in labor force',
                          '02': 'Living with two parents: Father only in labor force',
                          '03': 'Living with two parents: Mother only in labor force',
                          '04': 'Living with two parents: Neither parent in labor force',
                          '05': 'Living with father: Father in the labor force',
                          '06': 'Living with father: Father not in labor force',
                          '07': 'Living with mother: Mother in the labor force',
                          '08': 'Living with mother: Mother not in labor force'})

DIS_FEATURE = Feature('DIS', cat_dtype, "Disability recode",
                      name_extended="Disability status",
                      value_mapping={'01': 'With a disability',
                                     '02': 'Without a disability'})

# TODO(jpgard): complete mapping for this feature.
OCCP_FEATURE = Feature('OCCP', cat_dtype,
                       "Occupation recode for 2018 and later based on 2018 OCC codes",
                       name_extended="Occupation")

WKHP_FEATURE = Feature('WKHP', int,
                       "Usual hours worked per week past 12 months",
                       name_extended="Usual hours worked per week past 12 months")

RELP_FEATURE = Feature('RELP', cat_dtype, "Relationship",
                       name_extended="Relationship to reference person",
                       value_mapping={'00': 'Reference person',
                                      '01': 'Husband/wife',
                                      '02': 'Biological son or daughter',
                                      '03': 'Adopted son or daughter',
                                      '04': 'Stepson or stepdaughter',
                                      '05': 'Brother or sister',
                                      '06': 'Father or mother',
                                      '07': 'Grandchild', '08': 'Parent-in-law',
                                      '09': 'Son-in-law or daughter-in-law',
                                      '10': 'Other relative',
                                      '11': 'Roomer or boarder',
                                      '12': 'Housemate or roommate',
                                      '13': 'Unmarried partner',
                                      '14': 'Foster child',
                                      '15': 'Other nonrelative',
                                      '16': 'Institutionalized group quarters population',
                                      '17': 'Noninstitutionalized group quarters population'})

POBP_FEATURE = Feature('POBP', cat_dtype, "Place of birth (Recode)",
                       name_extended="Place of birth")

ENG_FEATURE = Feature('ENG', cat_dtype, "Ability to speak English",
                      name_extended="Ability to speak English",
                      value_mapping={
                          '00': "less than 5 years old/speaks only English",
                          '01': "Very well", '02': "Well", '03': "Not well",
                          '04': "Not at all"})

MIG_FEATURE = Feature('MIG', cat_dtype,
                      "Mobility status (lived here 1 year ago)",
                      name_extended="Mobility status",
                      value_mapping={'00': 'N/A (less than 1 year old)',
                                     '01': 'Yes, same house(nonmovers)',
                                     '02': 'No, outside US and Puerto Rico',
                                     '03': 'No, different house in US or Puerto Rico'})
NATIVITY_FEATURE = Feature('NATIVITY', cat_dtype, "Nativity",
                           name_extended="Nativity",
                           value_mapping={'01': 'Native', '02': 'Foreign born'})
DEAR_FEATURE = Feature('DEAR', cat_dtype, "Hearing difficulty",
                       name_extended="Hearing difficulty",
                       value_mapping={'01': 'Yes', '02': 'No'})

################################################################################
# Various ACS-related constants
################################################################################


ACS_YEARS = [2014, 2015, 2016, 2017, 2018]

# Region codes; see 'DIVISION' feature below.
ACS_REGIONS = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09']

ACS_STATE_LIST = [
    'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI',
    'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI',
    'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC',
    'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT',
    'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'PR']

# copied from (non-importable) location in folktables.load_acs
_STATE_CODES = {'AL': '01', 'AK': '02', 'AZ': '04', 'AR': '05', 'CA': '06',
                'CO': '08', 'CT': '09', 'DE': '10', 'FL': '12', 'GA': '13',
                'HI': '15', 'ID': '16', 'IL': '17', 'IN': '18', 'IA': '19',
                'KS': '20', 'KY': '21', 'LA': '22', 'ME': '23', 'MD': '24',
                'MA': '25', 'MI': '26', 'MN': '27', 'MS': '28', 'MO': '29',
                'MT': '30', 'NE': '31', 'NV': '32', 'NH': '33', 'NJ': '34',
                'NM': '35', 'NY': '36', 'NC': '37', 'ND': '38', 'OH': '39',
                'OK': '40', 'OR': '41', 'PA': '42', 'RI': '44', 'SC': '45',
                'SD': '46', 'TN': '47', 'TX': '48', 'UT': '49', 'VT': '50',
                'VA': '51', 'WA': '53', 'WV': '54', 'WI': '55', 'WY': '56',
                'PR': '72'}

# Maps 2-digit numeric strings ('01') to
# human-readable 2-letter state names ('WA')
ST_CODE_TO_STATE = {v: k for k, v in _STATE_CODES.items()}

################################################################################
# Feature lists
################################################################################

# Set of features shared by all ACS tasks.
ACS_SHARED_FEATURES = FeatureList(features=[
    Feature('AGEP', int, "Age", name_extended='age in years'),
    Feature('SEX', int, "Sex",
            name_extended='sex',
            value_mapping={
                1: "Male", 2: "Female",
            }),
    Feature('ST', cat_dtype, "State Code based on 2010 Census definitions.",
            name_extended="State"),
    Feature('MAR', cat_dtype, "Marital status",
            name_extended='marital status',
            value_mapping={
                1: 'Married',
                2: 'Widowed',
                3: 'Divorced',
                4: 'Separated',
                5: 'Never married or under 15 years old'
            }),
    Feature('CIT', cat_dtype, """Citizenship status""",
            name_extended='citizenship status',
            value_mapping={
                1: 'Born in the U.S.',
                2: 'Born in Puerto Rico, Guam, the U.S. Virgin Islands, '
                   'or the Northern Marianas',
                3: 'Born abroad of American parent(s)',
                4: 'U.S. citizen by naturalization',
                5: 'Not a citizen of the U.S.',
            }),
    Feature('RAC1P', int, """Recoded detailed race code""",
            name_extended='race',
            value_mapping={
                1: 'White alone',
                2: 'Black or African American alone',
                3: 'American Indian alone',
                4: 'Alaska Native alone',
                5: 'American Indian and Alaska Native tribes specified; or'
                   ' American Indian or Alaska Native, not specified and '
                   'no other races',
                6: 'Asian alone',
                7: 'Native Hawaiian and Other Pacific Islander alone',
                8: 'Some Other Race alone',
                9: 'Two or More Races'}),
    Feature('SCHL', cat_dtype, "Educational attainment",
            name_extended="Educational attainment",
            value_mapping={
                np.nan: 'NA (less than 3 years old)',
                1: 'No schooling completed',
                2: 'Nursery school, preschool',
                3: 'Kindergarten',
                4: 'Grade 1',
                5: 'Grade 2',
                6: 'Grade 3',
                7: 'Grade 4',
                8: 'Grade 5',
                9: 'Grade 6',
                10: 'Grade 7',
                11: 'Grade 8',
                12: 'Grade 9',
                13: 'Grade 10',
                14: 'Grade 11',
                15: '12th grade - no diploma',
                16: 'Regular high school diploma',
                17: 'GED or alternative credential',
                18: 'Some college, but less than 1 year',
                19: '1 or more years of college credit, no degree',
                20: "Associate's degree",
                21: "Bachelor's degree",
                22: "Master's degree",
                23: "Professional degree beyond a bachelor's degree",
                24: 'Doctorate degree',
            }),
    Feature('DIVISION', cat_dtype,
            "Division code based on 2010 Census definitions.",
            name_extended='geographic region',
            value_mapping={
                0: 'Puerto Rico',
                1: 'New England (Northeast region)',
                2: 'Middle Atlantic (Northeast region)',
                3: 'East North Central (Midwest region)',
                4: 'West North Central (Midwest region)',
                5: 'South Atlantic (South region)',
                6: 'East South Central (South region)',
                7: 'West South Central (South Region)',
                8: 'Mountain (West region)',
                9: 'Pacific (West region)',
            }),
    Feature('ACS_YEAR', int, 'Derived feature for ACS year.',
            name_extended='Year of survey')],
    documentation="https://www2.census.gov/programs-surveys/acs/tech_docs"
                  "/pums/data_dict/PUMS_Data_Dictionary_2019.pdf"
)

ACS_INCOME_FEATURES = FeatureList([
    Feature('COW', cat_dtype, """Class of worker.""",
            name_extended='class of worker',
            value_mapping={
                '01': "Employee of a private for-profit company or business, "
                      "or of an individual, for wages, salary, or commissions",
                '02': "Employee of a private not-for-profit, tax-exempt, or charitable organization",
                '03': "Local government employee (city, county, etc.)",
                '04': "State government employee",
                '05': "Federal government employee",
                '06': "Self-employed in own not incorporated business, professional practice, or farm",
                '07': "Self-employed in own incorporated business, professional practice or farm",
                '08': "Working without pay in family business or farm"}),
    ENG_FEATURE,
    FER_FEATURE,
    Feature('HINS1', cat_dtype,
            "Insurance through a current or former employer or union",
            name_extended="Has health insurance through a current or former "
                          "employer or union",
            value_mapping={'01': 'Yes', '02': 'No'}),

    Feature('HINS2', cat_dtype,
            "Insurance purchased directly from an insurance company",
            name_extended="Has insurance purchased directly from an insurance company",
            value_mapping={'01': 'Yes', '02': 'No'}),
    Feature('HINS3', cat_dtype,
            "Medicare, for people 65 and older, or people with certain disabilities",
            name_extended="Has medicare",
            value_mapping={'01': 'Yes', '02': 'No'}),
    Feature('HINS4', cat_dtype, """Medicaid, Medical Assistance, or any kind 
    of government-assistance plan for those with low incomes or a disability""",
            name_extended="Has Medicaid, medical assistance, or any kind of "
                          "government-assistance plan for those with low "
                          "incomes or a disability",
            value_mapping={'01': 'Yes', '02': 'No'}),
    NWLA_FEATURE,
    NWLK_FEATURE,
    OCCP_FEATURE,
    POBP_FEATURE,
    RELP_FEATURE,
    WKHP_FEATURE,
    Feature('WKW', int, "Weeks worked during past 12 months.",
            name_extended="Weeks worked during past 12 months"),
    Feature('WRK', cat_dtype, "Worked last week",
            name_extended="Worked last week",
            value_mapping={'00': 'NA (not reported)',
                           '01': 'Worked',
                           '02': 'Did not work'}),
    Feature('PINCP', float, """Total person's income >= threshold.""",
            is_target=True),
],
    documentation="https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/PUMS_Data_Dictionary_2014-2018.pdf")

ACS_PUBCOV_FEATURES = FeatureList(features=[
    DIS_FEATURE,
    ESP_FEATURE,
    MIG_FEATURE,
    ANC_FEATURE,
    NATIVITY_FEATURE,
    DEAR_FEATURE,
    DEYE_FEATURE,
    DREM_FEATURE,
    Feature('PINCP', float, "Total person's income",
            name_extended="Total person's income in dollars"),
    Feature('ESR', cat_dtype, """Employment status recode b .N/A (less than 
    16 years old) 1 .Civilian employed, at work 2 .Civilian employed, with a 
    job but not at work 3 .Unemployed 4 .Armed forces, at work 5 .Armed 
    forces, with a job but not at work 6 .Not in labor force""",
            name_extended="Employment status",
            value_mapping={
                '00': 'N/A (less than 16 years old)',
                '01': 'Civilian employed, at work',
                '02': 'Civilian employed, with a job but not at work',
                '03': 'Unemployed',
                '04': 'Armed forces, at work',
                '05': 'Armed forces, with a job but not at work',
                '06': 'Not in labor force'}),
    FER_FEATURE,
    Feature('PUBCOV', int, """Public health coverage recode =With public 
    health coverage 0=Without public health coverage""", is_target=True)],
    documentation="https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/PUMS_Data_Dictionary_2014-2018.pdf")

ACS_UNEMPLOYMENT_FEATURES = FeatureList(features=[
    Feature('ESR', int, "Employment status (is unemployed)", is_target=True),
    ENG_FEATURE,
    POBP_FEATURE,
    RELP_FEATURE,
    WKHP_FEATURE,
    Feature('WKW', int, "Weeks worked during past 12 months",
            name_extended="Weeks worked during past 12 months"),
    Feature('WRK', cat_dtype, "Worked last week",
            name_extended="Worked last week",
            value_mapping={
                '00': 'N/A (not reported',
                '01': 'Worked',
                '02': 'Did not work'}),
    OCCP_FEATURE,
    DIS_FEATURE,
    ESP_FEATURE,
    MIG_FEATURE,
    MIL_FEATURE,
    ANC_FEATURE,
    NATIVITY_FEATURE,
    DEAR_FEATURE,
    DEYE_FEATURE,
    DREM_FEATURE,
    Feature('DPHY', cat_dtype, "Ambulatory difficulty",
            name_extended="Ambulatory difficulty",
            value_mapping={'00': 'N/A (Less than 5 years old)',
                           '01': 'Yes',
                           '02': 'No', }),
    FER_FEATURE,
],
    documentation="https://www2.census.gov/programs-surveys/acs/tech_docs"
                  "/pums/data_dict/PUMS_Data_Dictionary_2019.pdf"
)

ACS_FOODSTAMPS_FEATURES = FeatureList(features=[
    Feature('FS', int, """Yearly food stamp/Supplemental Nutrition Assistance 
    Program (SNAP) recipiency (household) b .N/A (vacant) 5 1 .Yes 2 .No""",
            is_target=True),
    ENG_FEATURE,
    FER_FEATURE,
    Feature('HUPAC', int, "Household presence and age of children",
            name_extended="Household presence and age of children",
            value_mapping={
                '00': 'N/A (GQ/vacant)',
                '01': 'With children under 6 years only',
                '02': 'With children 6 to 17 years only',
                '03': 'With children under 6 years and 6 to 17 years',
                '04': 'No children'
            }),
    Feature('WIF', int, "Workers in family during the past 12 months",
            name_extended="Workers in family during the past 12 months",
            value_mapping={
                0: 'No workers',
                1: '1 worker',
                2: '2 workers',
                3: '3 or more workers'}),
    NWLA_FEATURE,
    NWLK_FEATURE,
    OCCP_FEATURE,
    POBP_FEATURE,
    RELP_FEATURE,
    WKHP_FEATURE,
    Feature('WKW', int,
            "Weeks worked during past 12 months.",
            name_extended="Weeks worked during past 12 months",
            ),
    Feature('WRK', cat_dtype, "Worked last week",
            name_extended="worked last week",
            value_mapping={'01': 'worked', '02': 'did not work'}),
    DIS_FEATURE,
    MIL_FEATURE,
    ANC_FEATURE,
    NATIVITY_FEATURE,
    DEAR_FEATURE,
    DEYE_FEATURE,
    DREM_FEATURE,
    Feature('PUBCOV', cat_dtype, "Public health coverage recode",
            value_mapping={
                '00': 'Without public health coverage',
                '01': 'With public health coverage'})
],
    documentation="https://www2.census.gov/programs-surveys/acs/tech_docs"
                  "/pums/data_dict/PUMS_Data_Dictionary_2019.pdf"
)


################################################################################
# Preprocessing functions
################################################################################


def map_categorical_features(df, feature_mapping):
    """Convert a subset of features from numeric to categorical format.

    Note that this only maps a known set of features used in this work;
    there are likely many additional categorical features treated as numeric
    that could be returned by folktables!
    """
    for feature, mapping in feature_mapping.items():
        if feature in df.columns:
            assert pd.isnull(
                df[feature]).values.sum() == 0, "nan values in input"

            mapped_setdiff = set(df[feature].unique().tolist()) - set(
                list(mapping.keys()))
            assert not mapped_setdiff, "missing keys {} from mapping {}".format(
                list(mapped_setdiff), list(mapping.keys()))
            for x in df[feature].unique():
                try:
                    assert x in list(mapping.keys())
                except AssertionError:
                    raise ValueError(f"features {feature} value {x} not in "
                                     f"mapping keys {list(mapping.keys())}")
            if feature in df.columns:
                df[feature] = pd.Categorical(
                    df[feature].map(mapping),
                    categories=list(set(mapping.values())))
            assert pd.isnull(df[feature]).values.sum() == 0, \
                "nan values in output; check for non-mapped input values."
    return df


def acs_data_to_df(
        features: np.ndarray, label: np.ndarray,
        feature_list: FeatureList,
        feature_mapping: dict) -> pd.DataFrame:
    """
    Build a DataFrame from the result of folktables.BasicProblem.df_to_numpy().
    """
    ary = np.concatenate((features, label.reshape(-1, 1),), axis=1)
    df = pd.DataFrame(ary,
                      columns=feature_list.predictors + [feature_list.target])
    df = map_categorical_features(df, feature_mapping=feature_mapping)
    return df


def default_acs_group_transform(x):
    # 'White alone' vs. all other categories (RAC1P) or
    # 'Male' vs. Female (SEX)
    # Note that *privileged* group is coded as 1, by default.
    return x == 1


def default_acs_postprocess(x):
    return np.nan_to_num(x, -1)


def income_cls_target_transform(y, threshold):
    """Binarization target transform for income."""
    return y > threshold


def pubcov_target_transform(y, threshold):
    """Default Public Coverage target transform from folktables."""
    del threshold
    return y == 1


def unemployment_target_transform(y, threshold):
    """Default Public Coverage target transform from folktables."""
    del threshold
    return y == 3


def unemployment_filter(data):
    """
    Filters for the unemployment; focus on Americans of working age not eligible for Social Security.
    """
    df = data
    df = df[(df['AGEP'] < 62) & (df['AGEP'] >= 18)]
    return df


def foodstamps_target_transform(y, threshold):
    del threshold
    return y == 1


def foodstamps_filter(data):
    """Filter for food stamp recipiency task; focus on low income Americans (
    as in public coverage task) of working age (since these would be
    individuals actually pplying for this benefit, as opposed to children
    living in a household that receives food stamps). """
    df = data
    df = df[df['HUPAC'] >= 1]  # at least one child in household
    df = df[(df['AGEP'] < 62) & (df['AGEP'] >= 18)]
    return df[df['PINCP'] <= 30000]


@dataclass
class ACSTaskConfig:
    """A class to configure data loading/preprocessing for an ACS task."""
    features_to_use: FeatureList
    group_transform: Callable
    postprocess: Callable
    preprocess: Callable
    target: str
    target_transform: Callable
    threshold: Union[int, float]


ACS_TASK_CONFIGS = frozendict.frozendict({
    'income': ACSTaskConfig(**{
        'features_to_use': ACS_INCOME_FEATURES + ACS_SHARED_FEATURES,
        'group_transform': default_acs_group_transform,
        'postprocess': default_acs_postprocess,
        'preprocess': folktables.acs.adult_filter,
        'target': 'PINCP',
        'target_transform': income_cls_target_transform,
        'threshold': 56000,
    }),
    'pubcov': ACSTaskConfig(**{
        'features_to_use': ACS_PUBCOV_FEATURES + ACS_SHARED_FEATURES,
        'group_transform': default_acs_group_transform,
        'postprocess': default_acs_postprocess,
        'preprocess': folktables.acs.public_coverage_filter,
        'target': 'PUBCOV',
        'target_transform': pubcov_target_transform,
        'threshold': None,
    }),
    'unemployment': ACSTaskConfig(**{
        'features_to_use': ACS_UNEMPLOYMENT_FEATURES + ACS_SHARED_FEATURES,
        'group_transform': default_acs_group_transform,
        'postprocess': default_acs_postprocess,
        'preprocess': unemployment_filter,
        'target': 'ESR',
        'target_transform': unemployment_target_transform,
        'threshold': None,
    }),
    'foodstamps': ACSTaskConfig(**{
        'features_to_use': ACS_FOODSTAMPS_FEATURES + ACS_SHARED_FEATURES,
        'group_transform': default_acs_group_transform,
        'postprocess': default_acs_postprocess,
        'preprocess': foodstamps_filter,
        'target': 'FS',
        'target_transform': foodstamps_target_transform,
        'threshold': None,
    })
})


def get_acs_data_source(year, root_dir='datasets/acs'):
    return folktables.ACSDataSource(survey_year=str(year),
                                    horizon='1-Year',
                                    survey='person',
                                    root_dir=root_dir)


def preprocess_acs(df: pd.DataFrame):
    if 'ST' in df.columns:
        # Map numeric state codes to human-readable values
        df['ST'] = df['ST'].map(ST_CODE_TO_STATE)
        assert pd.isnull(df['ST']).sum() == 0
    return df

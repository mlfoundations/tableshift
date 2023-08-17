"""
NHANES-related tools. See also the documentation at the link below:
https://www.cdc.gov/Nchs/Nhanes/about_nhanes.htm

NHANES is a public data source and no special action is required
to access it.

For more information on datasets and access in TableShift, see:
* https://tableshift.org/datasets.html
* https://github.com/mlfoundations/tableshift

"""
import json
import logging
import os
from collections import defaultdict

import numpy as np
import pandas as pd

from tableshift.core.features import Feature, FeatureList, cat_dtype

DEFAULT_NHANES_CODING = {
    "1.0": "Yes",
    "2.0": "No",
    "7.0": "Refused",
    "9.0": "Don't know",
}

NHANES_YEARS = [1999, 2001, 2003, 2005, 2007, 2009, 2011, 2013, 2015, 2017]

# Dictionary mapping years to data sources. Because NHANES uses the same
# name for each file, we need to manually track the year associated with
# each dataset.
NHANES_DATA_SOURCES = os.path.join(os.path.dirname(__file__),
                                   "nhanes_data_sources.json")

# Mapping of NHANES component types to names of data sources to use. See
# nhanes_data_sources.json. This ensures that only needed files are
# downloaded/read from disk, because NHANES contains a huge number of sources
# per year.

NHANES_CHOLESTEROL_DATA_SOURCES_TO_USE = {
    "Demographics": [
        "Demographic Variables & Sample Weights",  # 1999 - 2003
        "Demographic Variables and Sample Weights"],  # 2005- 2017
    "Questionnaire": ["Blood Pressure & Cholesterol",  # All years
                      "Cardiovascular Health",  # All years
                      "Diabetes",  # All years
                      "Kidney Conditions",  # 1999
                      "Kidney Conditions - Urology",  # 2001 - 2017
                      "Medical Conditions",  # All years
                      "Osteoporosis",  # Not preset in 2011, 2015
                      ],
    "Laboratory": [
        "Cholesterol - LDL & Triglycerides",
        # 1999 - 2003, 2007 - 2013
        "Cholesterol - LDL, Triglyceride & Apoliprotein (ApoB)",
        # 2005
        "Cholesterol - Low - Density Lipoprotein (LDL) & Triglycerides",
        # 2015
        "Cholesterol - Low-Density Lipoproteins (LDL) & Triglycerides"
        # 2017
    ],
}

NHANES_LEAD_DATA_SOURCES_TO_USE = {
    "Demographics": [
        "Demographic Variables & Sample Weights",  # 1999 - 2003
        "Demographic Variables and Sample Weights"],  # 2005- 2017
    "Questionnaire": ["Diet Behavior & Nutrition",
                      # Note: prior to 2017 income questions are in
                      # Demographics file.
                      "Income"
                      ],
    "Laboratory": [
        "Cadmium, Lead, Mercury, Cotinine & Nutritional Biochemistries",  # 1999
        "Cadmium, Lead, Total Mercury, Ferritin, Serum Folate, RBC Folate, "
        "Vitamin B12, Homocysteine, Methylmalonic "
        "acid, Cotinine - Blood, Second Exam",  # 2001
        "Cadmium, Lead, & Total Mercury - Blood",  # 2003
        "Cadmium, Lead, & Total Mercury - Blood",  # 2005
        "Cadmium, Lead, & Total Mercury - Blood",  # 2007
        "Cadmium, Lead, & Total Mercury - Blood",  # 2009
        "Cadmium, Lead, Total Mercury, Selenium, & Manganese - Blood",  # 2011
        "Lead, Cadmium, Total Mercury, Selenium, and Manganese - Blood",  # 2013
        "Lead, Cadmium, Total Mercury, Selenium & Manganese - Blood",  # 2015
        "Lead, Cadmium, Total Mercury, Selenium, & Manganese - Blood"]  # 2017
}


def get_nhanes_data_sources(task: str, years=None):
    """Fetch a mapping of {year: list of urls} for NHANES."""
    years = [int(x) for x in years]
    if task == "cholesterol":
        data_sources_to_use = NHANES_CHOLESTEROL_DATA_SOURCES_TO_USE
    elif task == "lead":
        data_sources_to_use = NHANES_LEAD_DATA_SOURCES_TO_USE
    else:
        raise ValueError

    output = defaultdict(list)
    with open(NHANES_DATA_SOURCES, "r") as f:
        data_sources = json.load(f)
    for year, components in data_sources.items():
        if (years is not None) and (int(year) in years):
            for component, sources in components.items():
                for source_name, source_url in sources.items():
                    if source_name in data_sources_to_use[component]:
                        output[year].append(source_url)
    return output


NHANES_CHOLESTEROL_FEATURES = FeatureList(features=[

    Feature('LBDLDL', float, is_target=True,
            description='Direct LDL-Cholesterol (mg/dL)'),

    # Below we use the additional set of risk factors listed in the above report
    # (Table 6) **which can be asked in a questionnaire** (i.e. those which
    # do not require laboratory testing).

    ####### Risk Factor: Family history of ASCVD

    # No questions on this topic.

    ####### Risk Factor: Metabolic syndrome (increased waist circumference,
    # elevated triglycerides [>175 mg/dL], elevated blood pressure,
    # elevated glucose, and low HDL-C [<40 mg/dL in men; <50 in women
    # mg/dL] are factors; tally of 3 makes the diagnosis)

    Feature('BPQ020', cat_dtype, """{Have you/Has SP} ever been told by a 
    doctor or other health professional # that {you/s/he} had hypertension, 
    also called high blood pressure?""",
            name_extended='ever been told by a doctor or other health professional that you had hypertension',
            value_mapping=DEFAULT_NHANES_CODING),

    Feature('DIQ160', cat_dtype, """{Have you/Has SP} ever been told by a 
    doctor or other health professional that {you have/SP has} any of the 
    following: prediabetes, impaired fasting glucose, impaired glucose 
    tolerance, borderline diabetes or that {your/her/his} blood sugar is 
    higher than normal but not high enough to be called diabetes or sugar 
    diabetes?""",
            name_extended="ever been told by a "
                          "doctor or other health professional that {you have/SP has} any of the "
                          "following: prediabetes, impaired fasting glucose, impaired glucose "
                          "tolerance, borderline diabetes or that {your/her/his} blood sugar is "
                          "higher than normal but not high enough to be called diabetes or sugar "
                          "diabetes?,",
            value_mapping=DEFAULT_NHANES_CODING),

    Feature('DIQ010', cat_dtype, """{Other than during pregnancy, {have 
    you/has SP}/{Have you/Has SP}} ever been told by a doctor or health 
    professional that {you have/{he/she/SP} has} diabetes or sugar 
    diabetes?""",
            name_extended="{Other than during pregnancy, have you"
                          " ever been told by a doctor or health "
                          "professional that you have diabetes or sugar diabetes?",
            value_mapping={**DEFAULT_NHANES_CODING,
                           "3.0": "Borderline"}),

    ####### Risk Factor: Chronic kidney disease

    Feature('KIQ025', cat_dtype, """In the past 12 months, {have you/has SP} 
    received dialysis (either hemodialysis or peritoneal dialysis)?""",
            name_extended="received dialysis (either hemodialysis or "
                          "peritoneal dialysis) in the past 12 months",
            value_mapping=DEFAULT_NHANES_CODING),

    Feature('KIQ022', cat_dtype, """{Have you/Has SP} ever been told by a 
    doctor or other health professional that {you/s/he} had weak or failing 
    kidneys? Do not include kidney stones, bladder infections, 
    or incontinence.""",
            name_extended="ever been told by a doctor or other health "
                          "professional that you had weak or failing kidneys",
            value_mapping=DEFAULT_NHANES_CODING),

    ####### Risk Factor: Chronic inflammatory conditions such as
    # psoriasis, RA, or HIV/AIDS

    Feature('MCQ070', cat_dtype,
            description="{Have you/Has SP} ever been told by a doctor or "
                        "other health care professional that {you/s/he} had "
                        "psoriasis ( sore-eye-asis)? (note: not present after "
                        "2013)",
            name_extended="ever been told by a doctor or other health care "
                          "professional that you had psoriasis",
            value_mapping=DEFAULT_NHANES_CODING),

    Feature('MCQ160A', cat_dtype, """Has a doctor or other health 
    professional ever told {you/SP} that {you/s/he} . . .had arthritis (
    ar-thry-tis)?""",
            name_extended="ever been told by a doctor or other health care "
                          "professional that you had arthritis",
            value_mapping=DEFAULT_NHANES_CODING),

    # Note: no questions about HIV/AIDS.

    #######  Risk Factor: History of premature menopause (before age 40 y)
    # and history of pregnancy-associated conditions that increase later
    #  ASCVD risk such as preeclampsia

    # Note: no questions on these.

    # #######  Risk Factor: High-risk race/ethnicities (eg South Asian ancestry)
    # Covered in shared 'RIDRETH' feature
], documentation="https://wwwn.cdc.gov/Nchs/Nhanes/")

NHANES_LEAD_FEATURES = FeatureList(features=[

    # A ratio of family income to poverty guidelines.
    Feature('INDFMPIRBelowCutoff', float,
            'Binary indicator for whether family PIR (poverty-income ratio)'
            'is <= 1.3. The threshold of 1.3 is selected based on the '
            'categorization in NHANES, where PIR <= 1.3 is the lowest level ('
            'see INDFMMPC feature).',
            name_extended='Binary indicator for whether family PIR ('
                          'poverty-income ratio) is <= 1.3.',
            value_mapping={1.: 'yes', 0.: 'no'}),

    Feature("LBXBPB", float, "Blood lead (ug/dL)", is_target=True,
            na_values=(".",)),
], documentation="https://wwwn.cdc.gov/Nchs/Nhanes/")

NHANES_SHARED_FEATURES = FeatureList(features=[
    # Derived feature for survey year
    Feature("nhanes_year", int, "Derived feature for year.",
            name_extended='year'),

    Feature('DMDBORN4', cat_dtype, """In what country {were you/was SP} born? 
    1	Born in 50 US states or Washington, DC 2 Others""",
            na_values=(77, 99, "."),
            name_extended="country of birth",
            value_mapping={"1.0": "born in 50 US states or Washington, DC",
                           "2.0": "not born in 50 US states or Washington, DC"}),

    Feature('DMDEDUC2', cat_dtype, """What is the highest grade or level of 
    school {you have/SP has} completed or the highest degree {you have/s/he 
    has} received?""",
            name_extended="highest grade or level of school completed or "
                          "highest degree received"),

    Feature('RIDAGEYR', float, """Age in years of the participant at the time 
    of screening. Individuals 80 and over are topcoded at 80 years of age.""",
            name_extended="age in years"),

    Feature('RIAGENDR', cat_dtype, "Gender of the participant.",
            name_extended="gender"),

    Feature('DMDMARTL', cat_dtype, "Marital status",
            name_extended="marital status"),

    Feature('RIDRETH_merged', int, """Derived feature. This feature uses 
    'RIDRETH3' (Recode of reported race and Hispanic origin information, 
    with Non-Hispanic Asian Category) from years where it is available, 
    and otherwise 'RIDRETH1' (Recode of reported race and Hispanic origin 
    information). 'RIDRETH3' contains a superset of the values in 'RIDRETH1' 
    but the shared values are coded identically; 'RIDRETH3' was only added to 
    NHANES in 2011-2012 data year. See _merge_ridreth_features( ) below, 
    and the NHANES documentation, e.g. 
    https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/DEMO_J.htm#RIDRETH1 .""",
            name_extended="race and hispanic origin",
            value_mapping={
                1.: "Mexican American",
                2.: "Other Hispanic",
                3.: "Non-Hispanic White",
                4.: "Non-Hispanic Black",
                5.: "Other Race - Including Multi-Racial",
                6.: "Non-Hispanic Asian",
                7.: "Other Race - Including Multi-Racial"
            }),

], documentation="https://wwwn.cdc.gov/Nchs/Nhanes/")


def _postprocess_nhanes(df: pd.DataFrame,
                        feature_list: FeatureList) -> pd.DataFrame:
    # Fill categorical missing values with "missing".
    for feature in feature_list:
        name = feature.name
        if name not in df.columns:
            logging.warning(
                f"feature {feature.name} missing; filling with "
                f"indicator; this can happen when data is subset by years since"
                f" some questions are not asked in all survey years.")
            df[name] = pd.Series(["MISSING"] * len(df))

        elif name != feature_list.target and feature.kind == cat_dtype:
            logging.debug(f"filling and casting categorical feature {name}")
            df[name] = df[name].fillna("MISSING").apply(str).astype("category")

    df.reset_index(drop=True, inplace=True)
    return df


def _merge_ridreth_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create a single race/ethnicity feature by using 'RIDRETH3'
    where available, else 'RIDRETH1'. """
    if ('RIDRETH3' in df.columns) and ('RIDRETH1' in df.columns):
        race_col = np.where(~np.isnan(df['RIDRETH3']), df['RIDRETH3'],
                            df['RIDRETH1'])
        df.drop(columns=['RIDRETH3', 'RIDRETH1'], inplace=True)
    elif 'RIDRETH3' in df.columns:
        race_col = df['RIDRETH3']
        df.drop(columns=['RIDRETH3'], inplace=True)
    else:
        race_col = df['RIDRETH1']
        df.drop(columns=['RIDRETH1'], inplace=True)

    df['RIDRETH_merged'] = race_col
    return df


def preprocess_nhanes_cholesterol(df: pd.DataFrame, threshold=160.):
    feature_list = NHANES_CHOLESTEROL_FEATURES + NHANES_SHARED_FEATURES
    target = feature_list.target

    df = _merge_ridreth_features(df)

    # Drop observations with missing target or missing domain split variable
    df.dropna(subset=[target, 'RIDRETH_merged'], inplace=True)

    # Binarize the target
    df[target] = (df[target] >= threshold).astype(float)

    df = _postprocess_nhanes(df, feature_list=feature_list)
    return df


def preprocess_nhanes_lead(df: pd.DataFrame, threshold: float = 3.5):
    """Preprocess the NHANES lead prediction dataset.

    The value of 3.5 µg/dl is based on the CDC Blood Lead Reference Value
    (BLRF) https://www.cdc.gov/nceh/lead/prevention/blood-lead-levels.htm
    """
    feature_list = NHANES_LEAD_FEATURES + NHANES_SHARED_FEATURES
    target = NHANES_LEAD_FEATURES.target
    df = _merge_ridreth_features(df)

    # Drop observations with missing target and missing domain split
    df = df.dropna(subset=[target, 'INDFMPIR', 'RIDAGEYR'])

    # Keep only children
    df = df[df['RIDAGEYR'] <= 18.]

    # Create the domain split variable for poverty-income ratio
    df['INDFMPIRBelowCutoff'] = (df['INDFMPIR'] <= 1.3).astype(int)
    df.drop(columns=['INDFMPIR'])

    # Binarize the target
    df[target] = (df[target] >= threshold).astype(float)

    df = _postprocess_nhanes(df, feature_list=feature_list)
    return df

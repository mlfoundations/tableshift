import json
import os

import pandas as pd
from tableshift.core.features import Feature, FeatureList, cat_dtype

DIABETES_READMISSION_RESOURCES = [
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00296"
    "/dataset_diabetes.zip"
]

# Common feature description for the 24 features for medications
DIABETES_MEDICATION_FEAT_DESCRIPTION = """Indicates if there was any diabetic 
    medication prescribed. Values: 'yes' and 'no' For the generic names: 
    metformin, repaglinide, nateglinide, chlorpropamide, glimepiride, 
    acetohexamide, glipizide, glyburide, tolbutamide, pioglitazone, 
    rosiglitazone, acarbose, miglitol, troglitazone, tolazamide, examide, 
    sitagliptin, insulin, glyburide-metformin, glipizide-metformin, 
    glimepiride-pioglitazone, metformin-rosiglitazone, 
    and metformin-pioglitazone, the feature indicates whether the drug was 
    prescribed or there was a change in the dosage. Values: 'up' if the 
    dosage was increased during the encounter, 'down' if the dosage was 
    decreased, 'steady' if the dosage did not change, and 'no' if the drug 
    was not prescribed."""


def get_icd9(depth=3) -> dict:
    """Fetch a dictionary mapping ICD9 codes to string descriptors."""
    # via https://raw.githubusercontent.com/sirrice/icd9/master/codes.json
    fp = os.path.join(os.path.dirname(__file__), "./icd9-codes.json")
    with open(fp, "r") as f:
        raw = f.read()
    icd9_codes = json.loads(raw)
    depth_mapping = {c["code"]: c["descr"]
                     for group in icd9_codes for c in group
                     if c['depth'] == depth}
    return depth_mapping


# Note: the UCI version of this dataset does *not* exactly correspond to the
# version documented in the linked paper. For example, in the paper, 'weight'
# is described as a numeric feature, but is discretized into bins in UCI;
# similarly, many fields are marked as having much higher missing value counts
# in the paper than are present in the UCI data. The cause of this discrepancy
# is not clear.
DIABETES_READMISSION_FEATURES = FeatureList(features=[
    Feature('race', cat_dtype, """Nominal. Values: Caucasian, Asian, African 
    American, Hispanic, and other"""),
    Feature('gender', cat_dtype, """Nominal. Values: male, female, and 
    unknown/invalid."""),
    Feature('age', cat_dtype, """Nominal. Grouped in 10-year intervals: [0, 
    10), [10, 20), . . ., [90, 100)"""),
    Feature('weight', cat_dtype, "Weight in pounds. Grouped in 25-pound "
                                 "intervals."),
    Feature('admission_type_id', int, """Integer identifier corresponding 
    to 9 distinct values.""",
            value_mapping={
                1: 'Emergency', 2: 'Urgent', 3: 'Elective', 4: 'Newborn',
                5: 'Not Available', 6: 'NULL', 7: 'Trauma Center',
                8: 'Not Mapped',
            },
            name_extended="Admission type"),
    Feature(
        'discharge_disposition_id', int,
        "Integer identifier corresponding to 29 distinct values.",
        name_extended="Discharge type",
        value_mapping={
            1: 'Discharged to home',
            2: 'Discharged/transferred to another short term hospital',
            3: 'Discharged/transferred to SNF',
            4: 'Discharged/transferred to ICF',
            5: 'Discharged/transferred to another type of inpatient care institution',
            6: 'Discharged/transferred to home with home health service',
            7: 'Left AMA',
            8: 'Discharged/transferred to home under care of Home IV provider',
            9: 'Admitted as an inpatient to this hospital',
            10: 'Neonate discharged to  another hospital for neonatal aftercare',
            11: 'Expired',
            12: 'Still patient or expected to return for outpatient services',
            13: 'Hospice / home',
            14: 'Hospice / medical facility',
            15: 'Discharged/transferred within this institution to Medicare approved swing bed',
            16: 'Discharged/transferred/referred another institution for outpatient services',
            17: 'Discharged/transferred/referred to this institution for outpatient services',
            18: 'NULL',
            19: 'Expired at home. Medicaid only, hospice.',
            20: 'Expired in a medical facility. Medicaid only, hospice.',
            21: 'Expired: place unknown. Medicaid only, hospice.',
            22: 'Discharged/transferred to another rehab fac including rehab units of a hospital.',
            23: 'Discharged/transferred to a long term care hospital.',
            24: 'Discharged/transferred to a nursing facility certified under Medicaid  but not certified under Medicare.',
            25: 'Not Mapped',
            26: 'Unknown/Invalid',
            27: 'Discharged/transferred to a federal health care facility.',
            28: 'Discharged/transferred/referred to a psychiatric hospital of psychiatric distinct part unit of a hospital',
            29: 'Discharged/transferred to a Critical Access Hospital (CAH).',
            30: 'Discharged/transferred to another Type of Health Care Institution not Defined Elsewhere',
        }),
    Feature('admission_source_id', int, """Integer identifier corresponding 
    to 21 distinct values.""",
            name_extended="Admission source",
            value_mapping={
                1: 'Physician Referral', 2: 'Clinic Referral ',
                3: 'HMO Referral', 4: 'Transfer from a hospital',
                5: 'Transfer from a Skilled Nursing Facility (SNF)',
                6: 'Transfer from another health care facility',
                7: 'Emergency Room',
                8: 'Court/Law Enforcement',
                9: 'Not Available',
                10: 'Transfer from critial access hospital',
                11: 'Normal Delivery',
                12: 'Premature Delivery',
                13: 'Sick Baby', 14: 'Extramural Birth', 15: 'Not Available',
                17: 'NULL',
                18: 'Transfer From Another Home Health Agency',
                19: 'Readmission to Same Home Health Agency',
                20: 'Not Mapped', 21: 'Unknown/Invalid',
                22: 'Transfer from hospital inpt/same fac reslt in a sep claim',
                23: 'Born inside this hospital',
                24: 'Born outside this hospital',
                25: 'Transfer from Ambulatory Surgery Center',
                26: 'Transfer from Hospice',
            }),
    Feature('time_in_hospital', float,
            "Integer number of days between admission and discharge",
            name_extended="Count of days beween admission and discharge"),
    Feature('payer_code', cat_dtype, "Integer identifier corresponding to 23 "
                                     "distinct values, for example, "
                                     "Blue Cross\Blue Shield, Medicare, "
                                     "and self-pay",
            name_extended="Payer code"),
    Feature('medical_specialty', cat_dtype, "Specialty of the admitting "
                                            "physician, corresponding to 84 "
                                            "distinct values, for example, "
                                            "cardiology, internal medicine, "
                                            "family/general practice, "
                                            "and surgeon.",
            name_extended="Medical specialty of the admitting physician"),
    Feature('num_lab_procedures', float, "Number of lab tests performed "
                                         "during the encounter",
            name_extended="Number of lab tests performed during the encounter"),
    Feature('num_procedures', float, "Number of procedures (other than lab "
                                     "tests) performed during the encounter",
            name_extended="Number of procedures (other than lab tests) "
                          "performed during the encounter"),
    Feature('num_medications', float, "Number of distinct generic names "
                                      "administered during the encounter",
            name_extended="Number of distinct generic drugs "
                          "administered during the encounter"),
    Feature('number_outpatient', float, "Number of outpatient visits of the "
                                        "patient in the year preceding the "
                                        "encounter",
            name_extended="Number of outpatient visits of the "
                          "patient in the year preceding the "
                          "encounter"),
    Feature('number_emergency', float,
            "Number of emergency visits of the "
            "patient in the year preceding the "
            "encounter",
            name_extended="Number of emergency visits of the "
                          "patient in the year preceding the "
                          "encounter"),
    Feature('number_inpatient', float,
            "Number of inpatient visits of the "
            "patient in the year preceding the "
            "encounter",
            name_extended="Number of inpatient visits of the "
                          "patient in the year preceding the "
                          "encounter"),
    Feature('diag_1', cat_dtype, "The primary diagnosis (coded as first three "
                                 "digits of ICD9); 848 distinct values",
            value_mapping=get_icd9(),
            name_extended="Primary diagnosis",
            ),
    Feature('diag_2', cat_dtype, "Secondary diagnosis (coded as first three "
                                 "digits of ICD9); 923 distinct values",
            name_extended="Secondary diagnosis",
            value_mapping=get_icd9()),
    Feature('diag_3', cat_dtype, "Additional secondary diagnosis (coded as "
                                 "first three digits of ICD9); 954 distinct "
                                 "values",
            name_extended="Additional secondary diagnosis",
            value_mapping=get_icd9()),
    Feature('number_diagnoses', float, "Number of diagnoses entered to the "
                                       "system",
            name_extended="Total number of diagnoses"),
    Feature('max_glu_serum', cat_dtype, "Indicates the range of the result or "
                                        "if the test was not taken. Values: "
                                        "'>200,' '>300,' 'normal,' and 'none' "
                                        "if not measured",
            name_extended="Max glucose serum"),
    Feature('A1Cresult', cat_dtype, "Indicates the range of the result or if "
                                    "the test was not taken. Values: '>8' if "
                                    "the result was greater than 8%, '>7' if "
                                    "the result was greater than 7% but less "
                                    "than 8%, 'normal' if the result was less "
                                    "than 7%, and 'none' if not measured.",
            name_extended='Hemoglobin A1c test result',
            note="""From original citation: Hemoglobin A1c (HbA1c) is an 
            important measure of glucose control, which is widely applied to 
            measure performance of diabetes care. (See documentation link.)"""
            ),
    Feature('metformin', cat_dtype, "Indicates if there was a change in "
                                    "diabetic medications (either dosage or "
                                    "generic name). Values: 'change' and 'no "
                                    "change'",
            name_extended='Change in metformin medication'),
    Feature('repaglinide', cat_dtype, DIABETES_MEDICATION_FEAT_DESCRIPTION,
            name_extended='Change in repaglinide medication'),
    Feature('nateglinide', cat_dtype, DIABETES_MEDICATION_FEAT_DESCRIPTION,
            name_extended='Change in nateglinide medication'),
    Feature('chlorpropamide', cat_dtype, DIABETES_MEDICATION_FEAT_DESCRIPTION,
            name_extended='Change in chlorpropamide medication'),
    Feature('glimepiride', cat_dtype, DIABETES_MEDICATION_FEAT_DESCRIPTION,
            name_extended='Change in glimepiride medication'),
    Feature('acetohexamide', cat_dtype, DIABETES_MEDICATION_FEAT_DESCRIPTION,
            name_extended='Change in acetohexamide medication'),
    Feature('glipizide', cat_dtype, DIABETES_MEDICATION_FEAT_DESCRIPTION,
            name_extended='Change in glipizide medication'),
    Feature('glyburide', cat_dtype, DIABETES_MEDICATION_FEAT_DESCRIPTION,
            name_extended='Change in glyburide medication'),
    Feature('tolbutamide', cat_dtype, DIABETES_MEDICATION_FEAT_DESCRIPTION,
            name_extended='Change in tolbutamide medication'),
    Feature('pioglitazone', cat_dtype, DIABETES_MEDICATION_FEAT_DESCRIPTION,
            name_extended='Change in pioglitazone medication'),
    Feature('rosiglitazone', cat_dtype, DIABETES_MEDICATION_FEAT_DESCRIPTION,
            name_extended='Change in rosiglitazone medication'),
    Feature('acarbose', cat_dtype, DIABETES_MEDICATION_FEAT_DESCRIPTION,
            name_extended='Change in acarbose medication'),
    Feature('miglitol', cat_dtype, DIABETES_MEDICATION_FEAT_DESCRIPTION,
            name_extended='Change in miglitol medication'),
    Feature('troglitazone', cat_dtype, DIABETES_MEDICATION_FEAT_DESCRIPTION,
            name_extended='Change in troglitazone medication'),
    Feature('tolazamide', cat_dtype, DIABETES_MEDICATION_FEAT_DESCRIPTION,
            name_extended='Change in tolazamide medication'),
    Feature('examide', cat_dtype, DIABETES_MEDICATION_FEAT_DESCRIPTION,
            name_extended='Change in examide medication'),
    Feature('citoglipton', cat_dtype, DIABETES_MEDICATION_FEAT_DESCRIPTION,
            name_extended='Change in citoglipton medication'),
    Feature('insulin', cat_dtype, DIABETES_MEDICATION_FEAT_DESCRIPTION,
            name_extended='Change in insulin medication'),
    Feature('glyburide-metformin', cat_dtype,
            DIABETES_MEDICATION_FEAT_DESCRIPTION,
            name_extended='Change in glyburide-metformin medication'),
    Feature('glipizide-metformin', cat_dtype,
            DIABETES_MEDICATION_FEAT_DESCRIPTION,
            name_extended='Change in glipizide-metformin medication'),
    Feature('glimepiride-pioglitazone', cat_dtype,
            DIABETES_MEDICATION_FEAT_DESCRIPTION,
            name_extended='Change in glimepiride-pioglitazone medication'),
    Feature('metformin-rosiglitazone', cat_dtype,
            DIABETES_MEDICATION_FEAT_DESCRIPTION,
            name_extended='Change in metformin-rosiglitazone medication'),
    Feature('metformin-pioglitazone', cat_dtype,
            DIABETES_MEDICATION_FEAT_DESCRIPTION,
            name_extended='Change in metformin-pioglitazone medication'),
    Feature('change', cat_dtype, "Indicates if there was a change in diabetic "
                                 "medications (either dosage or generic "
                                 "name). Values: 'change' and 'no change'",
            name_extended="Change in any medication"),
    Feature('diabetesMed', cat_dtype, "Indicates if there was any diabetic "
                                      "medication prescribed. Values: 'yes' "
                                      "and 'no'",
            name_extended="Diabetes medication prescribed"),
    # Converted to binary (readmit vs. no readmit).
    Feature('readmitted', float, "30 days, '>30' if the patient was "
                                 "readmitted in more than 30 days, and 'No' "
                                 "for no record of readmission.",
            is_target=True),
], documentation="http://www.hindawi.com/journals/bmri/2014/781670/")


def preprocess_diabetes_readmission(df: pd.DataFrame):
    # Drop 2273 obs with missing race (2.2336% of total data)
    df.dropna(subset=["race"], inplace=True)

    tgt_col = DIABETES_READMISSION_FEATURES.target
    df[tgt_col] = (df[tgt_col] != "NO").astype(float)

    # Some columns contain a small fraction of missing values (~1%); fill them.
    df.fillna("MISSING")
    return df

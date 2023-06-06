import pandas as pd

from tableshift.core.features import Feature, FeatureList, cat_dtype

GERMAN_RESOURCES = [
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "statlog/german/german.data"
]

GERMAN_FEATURES = FeatureList(features=[
    Feature("status", cat_dtype,
            description="Status of existing checking account",
            name_extended="Status of existing checking account",
            value_mapping={
                "A11": "< 0 DM ",
                "A12": "0 <= ... < 200 DM ",
                "A13": "... >= 200 DM / salary assignments for at least 1 year ",
                "A14": "no checking account ",
            }),
    Feature("duration", float,
            description="Duration in month",
            name_extended="Duration in month"),
    Feature("credit_history", cat_dtype,
            description="Credit history ",
            name_extended="Applicant's credit history ",
            value_mapping={
                "A30": "no credits taken/ all credits paid back duly ",
                "A31": "all credits at this bank paid back duly ",
                "A32": "existing credits paid back duly till now",
                "A33": "delay in paying off in the past",
                "A34": "critical account/ other credits existing (not at this bank)",
            }),
    Feature("purpose", cat_dtype, description="Purpose",
            name_extended="Applicant's stated purpose of requested loan",
            value_mapping={
                "A40": "car (new)",
                "A41": "car (used)",
                "A42": "furniture/equipment",
                "A43": "radio/television",
                "A44": "domestic appliances",
                "A45": "repairs",
                "A46": "education",
                "A47": "vacation",
                "A48": "retraining",
                "A49": "business",
                "A410": "others"}),
    Feature("credit_amt", float, description="Credit amount",
            name_extended="Credit amount"),
    Feature("savings_acct_bonds", cat_dtype,
            description="Savings account/bonds",
            name_extended="Applicant's current total savings account/bonds",
            value_mapping={
                "A61": "... < 100 DM",
                "A62": "100 <= ... < 500 DM",
                "A63": "500 <= ... < 1000 DM",
                "A64": ".. >= 1000 DM ",
                "A65": "unknown/ no savings account"}),
    Feature("present_unemployed_since", cat_dtype,
            description="Present employment since ",
            name_extended="Duration applicant has held present employment",
            value_mapping={
                "A71": "unemployed",
                "A72": "less than 1 year",
                "A73": "at least 1 but less than 4 years",
                "A74": "at least 4 but less than 7 years",
                "A75": "at least 7 years "}),
    Feature("installment_rate", float,
            description="Installment rate in percentage of disposable income",
            name_extended="Installment rate in percentage of disposable income"),
    Feature("other_debtors", cat_dtype,
            description="Other debtors / guarantors",
            name_extended="Other debtors / guarantors"),
    Feature("pres_res_since", float,
            description="Present residence since",
            name_extended="Time applicant has resided in present residence"
            ),
    Feature("property", cat_dtype, description="Property",
            name_extended="Applicant's highest level of property owned",
            value_mapping={
                "A121": "real estate",
                "A122": "building society savings agreement/ life insurance",
                "A123": "car or other",
                "A124": "unknown / no property"}),
    Feature("age_geq_median", cat_dtype,
            description="Binary indicator for whether applicant's age is "
                        "greater than or equal to the median age of all "
                        "applicants",
            name_extended="Binary indicator for whether applicant's age is "
                          "greater than or equal to the median age of all "
                          "applicants",
            value_mapping={
                1: "greater than or equal to median age",
                0: "not greater than or equal to median age"
            }),
    Feature("sex", cat_dtype, value_mapping={1: "male", 0: "female"}),
    Feature("other_installment", cat_dtype,
            description="Other installment plans",
            name_extended="Other installment plans",
            value_mapping={
                'A141': 'bank',
                'A142': 'stores',
                'A143': 'none'}),
    Feature("housing", cat_dtype, description="Housing",
            name_extended="Current type of housing occupied",
            value_mapping={'A151': 'rent', 'A152': 'own', 'A153': 'for free'}),
    Feature("num_exist_credits", float,
            description="Number of existing credits at this bank ",
            name_extended="Number of existing credits at this bank"),
    Feature("job", cat_dtype,
            name_extended="Type of job",
            value_mapping={
                'A171': 'unemployed/ unskilled - non-resident',
                'A172': 'unskilled - resident',
                'A173': 'skilled employee / official',
                'A174': 'management/ self-employed/highly qualified employee/ '
                        'officer ',
            }),
    Feature("num_ppl", float,
            description="Number of people being liable to provide maintenance "
                        "for",
            name_extended="Number of people being liable to provide "
                          "maintenance for"),
    Feature("has_phone", cat_dtype,
            name_extended="Applicant has phone registered under their name",
            value_mapping={'A191': 'no', 'A192': 'yes'}),
    Feature("foreign_worker", cat_dtype,
            name_extended="Applicant is a foreign worker",
            value_mapping={'A201': 'yes', 'A202': 'no'}),
    Feature("Target", int, is_target=True)],
    documentation="https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)")


def preprocess_german(df: pd.DataFrame):
    df.columns = ["status", "duration", "credit_history",
                  "purpose", "credit_amt", "savings_acct_bonds",
                  "present_unemployed_since", "installment_rate",
                  "per_status_sex", "other_debtors", "pres_res_since",
                  "property", "age", "other_installment", "housing",
                  "num_exist_credits", "job", "num_ppl", "has_phone",
                  "foreign_worker", "Target"]
    # Code labels as in tfds; see
    # https://github.com/tensorflow/datasets/blob/master/"\
    # "tensorflow_datasets/structured/german_credit_numeric.py
    df["Target"] = 2 - df["Target"]
    # convert per_status_sex into separate columns.
    # Sex is 1 if male; else 0.
    df["sex"] = df["per_status_sex"].apply(
        lambda x: 1 if x not in ["A92", "A95"] else 0)
    # Age is 1 if above median age, else 0.
    median_age = df["age"].median()
    df["age_geq_median"] = df["age"].apply(lambda x: 1 if x > median_age else 0)

    df["single"] = df["per_status_sex"].apply(
        lambda x: 1 if x in ["A93", "A95"] else 0)

    df.drop(columns=["per_status_sex", "age"], inplace=True)
    return df

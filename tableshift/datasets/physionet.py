import pandas as pd

from tableshift.core.features import Feature, FeatureList, cat_dtype

PHYSIONET_FEATURES = FeatureList(features=[
    Feature("HR", float, name_extended="Heart rate (in beats per minute)"),
    Feature("O2Sat", float, name_extended="Pulse oximetry (%)"),
    Feature("Temp", float, name_extended="Temperature (deg C)"),
    Feature("SBP", float, name_extended="Systolic BP (mm Hg)"),
    Feature("MAP", float, name_extended="Mean arterial pressure (mm Hg)"),
    Feature("DBP", float, name_extended="Diastolic BP (mm Hg)"),
    Feature("Resp", float,
            name_extended="Respiration rate (breaths per minute)"),
    Feature("EtCO2", float, name_extended="End tidal carbon dioxide (mm Hg)"),
    Feature("BaseExcess", float, name_extended="Excess bicarbonate (mmol/L)"),
    Feature("HCO3", float, name_extended="Bicarbonate (mmol/L)"),
    Feature("FiO2", float, name_extended="Fraction of inspired oxygen (%)"),
    Feature("pH", float, name_extended="pH"),
    Feature("PaCO2", float,
            name_extended="Partial pressure of carbon dioxide from arterial "
                          "blood (mm Hg)"),
    Feature("SaO2", float,
            name_extended="Oxygen saturation from arterial blood (%)"),
    Feature("AST", float, name_extended="Aspartate transaminase (IU/L)"),
    Feature("BUN", float, name_extended="Blood urea nitrogen (mg/dL)"),
    Feature("Alkalinephos", float, name_extended="Alkaline phosphatase (IU/L)"),
    Feature("Calcium", float, name_extended="Calcium (mg/dL)"),
    Feature("Chloride", float, name_extended="Chloride (mmol/L)"),
    Feature("Creatinine", float, name_extended="Creatinine (mg/dL)"),
    Feature("Bilirubin_direct", float,
            name_extended="Direct bilirubin (mg/dL)"),
    Feature("Glucose", float, name_extended="Serum glucose (mg/dL)"),
    Feature("Lactate", float, name_extended="Lactic acid (mg/dL)"),
    Feature("Magnesium", float, name_extended="Magnesium (mmol/dL)"),
    Feature("Phosphate", float, name_extended="Phosphate (mg/dL)"),
    Feature("Potassium", float, name_extended="Potassium (mmol/L)"),
    Feature("Bilirubin_total", float, name_extended="Total bilirubin (mg/dL)"),
    Feature("TroponinI", float, name_extended="Troponin I (ng/mL)"),
    Feature("Hct", float, name_extended="Hematocrit (%)"),
    Feature("Hgb", float, name_extended="Hemoglobin (g/dL)"),
    Feature("PTT", float,
            name_extended="Partial thromboplastin time (seconds)"),
    Feature("WBC", float, name_extended="Leukocyte count (count/L)"),
    Feature("Fibrinogen", float,
            name_extended="Fibrinogen concentration (mg/dL)"),
    Feature("Platelets", float, name_extended="Platelet count (count/mL)"),
    Feature("Age", int, name_extended="Age (years)"),
    Feature("Gender", int, name_extended="Female (0) or male (1)"),
    Feature("Unit1", int,
            name_extended="Administrative identifier for ICU unit (MICU); "
                          "false (0) or true (1)"),
    Feature("Unit2", int,
            name_extended="Administrative identifier for ICU unit (SICU); "
                          "false (0) or true (1)"),
    Feature("HospAdmTime", float,
            name_extended="Time between hospital and ICU admission ("
                          "hours since ICU admission)"),
    Feature("ICULOS", float,
            name_extended="ICU length of stay (hours since ICU admission)"),
    Feature("SepsisLabel", int,
            name_extended="For septic patients, SepsisLabel is 1 if t ≥ "
                          "t_sepsis − 6 and 0 if t < t_sepsis − 6. For "
                          "non-septic patients, SepsisLabel is 0.",
            is_target=True),
    Feature("set", cat_dtype,
            "The training set (i..e hospital) from which an example is drawn "
            "unique (values: 'a', 'b').")
], documentation="https://physionet.org/content/challenge-2019/1.0.0"
                 "/physionet_challenge_2019_ccm_manuscript.pdf")


def preprocess_physionet(df: pd.DataFrame) -> pd.DataFrame:
    return df

"""
Contains task configurations.

A task is a set of features (including both predictors and a target variable)
along with a DataSource. Tasks are the fundamental benchmarks that comprise
the tableshift benchmark.
"""

from dataclasses import dataclass
from typing import Any, Dict
from .data_source import *
from .features import FeatureList

from tableshift.datasets import *


@dataclass
class TaskConfig:
    # The data_source_cls instantiates the DataSource,
    # which fetches data and preprocesses it using a preprocess_fn.
    data_source_cls: Any
    # The feature_list describes the schema of the data *after* the
    # preprocess_fn is applied. It is used to check the output of the
    # preprocess_fn, and features are dropped or type-cast as necessary.
    feature_list: FeatureList



# Mapping of task names to their configs. An arbitrary number of tasks
# can be created from a single data source, by specifying different
# preprocess_fn and features.
_TASK_REGISTRY: Dict[str, TaskConfig] = {
    "acsincome":
        TaskConfig(ACSDataSource,
                   ACS_INCOME_FEATURES + ACS_SHARED_FEATURES),
    "acsfoodstamps":
        TaskConfig(ACSDataSource,
                   ACS_FOODSTAMPS_FEATURES + ACS_SHARED_FEATURES),
    "acspubcov":
        TaskConfig(ACSDataSource,
                   ACS_PUBCOV_FEATURES + ACS_SHARED_FEATURES),
    "acsunemployment":
        TaskConfig(ACSDataSource,
                   ACS_UNEMPLOYMENT_FEATURES + ACS_SHARED_FEATURES),
    "adult":
        TaskConfig(AdultDataSource, ADULT_FEATURES),
    "anes":
        TaskConfig(ANESDataSource, ANES_FEATURES),
    "assistments":
        TaskConfig(AssistmentsDataSource, ASSISTMENTS_FEATURES),
    "brfss_diabetes":
        TaskConfig(BRFSSDataSource, BRFSS_DIABETES_FEATURES),
    "brfss_blood_pressure":
        TaskConfig(BRFSSDataSource, BRFSS_BLOOD_PRESSURE_FEATURES),
    "communities_and_crime":
        TaskConfig(CommunitiesAndCrimeDataSource, CANDC_FEATURES),
    "compas":
        TaskConfig(COMPASDataSource, COMPAS_FEATURES),
    "diabetes_readmission":
        TaskConfig(DiabetesReadmissionDataSource,
                   DIABETES_READMISSION_FEATURES),
    "german":
        TaskConfig(GermanDataSource, GERMAN_FEATURES),
    "heloc":
        TaskConfig(HELOCDataSource, HELOC_FEATURES),
    "metamimic_alcohol":
        TaskConfig(MetaMIMICDataSource, METAMIMIC_ALCOHOL_FEATURES),
    "metamimic_anemia":
        TaskConfig(MetaMIMICDataSource, METAMIMIC_ANEMIA_FEATURES),
    "metamimic_atrial":
        TaskConfig(MetaMIMICDataSource, METAMIMIC_ATRIAL_FEATURES),
    "metamimic_diabetes":
        TaskConfig(MetaMIMICDataSource, METAMIMIC_DIABETES_FEATURES),
    "metamimic_heart":
        TaskConfig(MetaMIMICDataSource, METAMIMIC_HEART_FEATURES),
    "metamimic_hypertension":
        TaskConfig(MetaMIMICDataSource, METAMIMIC_HYPERTENSIVE_FEATURES),
    "metamimic_hypotension":
        TaskConfig(MetaMIMICDataSource, METAMIMIC_HYPOTENSION_FEATURES),
    "metamimic_ischematic":
        TaskConfig(MetaMIMICDataSource, METAMIMIC_ISCHEMATIC_FEATURES),
    "metamimic_lipoid":
        TaskConfig(MetaMIMICDataSource, METAMIMIC_LIPOID_FEATURES),
    "metamimic_overweight":
        TaskConfig(MetaMIMICDataSource, METAMIMIC_OVERWEIGHT_FEATURES),
    "metamimic_purpura":
        TaskConfig(MetaMIMICDataSource, METAMIMIC_PURPURA_FEATURES),
    "metamimic_respiratory":
        TaskConfig(MetaMIMICDataSource, METAMIMIC_RESPIRATORY_FEATURES),
    "mimic_extract_los_3":
        TaskConfig(MIMICExtractDataSource, MIMIC_EXTRACT_LOS_3_FEATURES),
    "mimic_extract_los_3_selected":
        TaskConfig(MIMICExtractDataSource, MIMIC_EXTRACT_LOS_3_SELECTED_FEATURES),
    "mimic_extract_mort_hosp":
        TaskConfig(MIMICExtractDataSource, MIMIC_EXTRACT_MORT_HOSP_FEATURES),
    "mimic_extract_mort_hosp_selected":
        TaskConfig(MIMICExtractDataSource, MIMIC_EXTRACT_MORT_HOSP_SELECTED_FEATURES),
    "mooc":
        TaskConfig(MOOCDataSource, MOOC_FEATURES),
    "nhanes_cholesterol":
        TaskConfig(NHANESDataSource,
                   NHANES_CHOLESTEROL_FEATURES + \
                   NHANES_SHARED_FEATURES),
    "nhanes_lead":
        TaskConfig(NHANESDataSource,
                   NHANES_SHARED_FEATURES +
                   NHANES_LEAD_FEATURES),
    "physionet":
        TaskConfig(PhysioNetDataSource, PHYSIONET_FEATURES),
    "electricity":
        TaskConfig(GrinstajnHFDataSource, ELECTRICITY_FEATURES),
    "bank-marketing":
        TaskConfig(GrinstajnHFDataSource, BANK_MARKETING_FEATURES),
    "california":
        TaskConfig(GrinstajnHFDataSource, CALIFORNIA_FEATURES),
    'covertype':
        TaskConfig(GrinstajnHFDataSource, COVERTYPE_FEATURES),
    'credit':
        TaskConfig(GrinstajnHFDataSource, CREDIT_FEATURES),
    'default-of-credit-card-clients':
        TaskConfig(GrinstajnHFDataSource, DEFAULT_OF_CREDIT_CLIENTS_FEATURES),
    'eye_movements':
        TaskConfig(GrinstajnHFDataSource, EYE_MOVEMENTS_FEATURES),
    'Higgs':
        TaskConfig(GrinstajnHFDataSource, HIGGS_FEATURES),
    'MagicTelescope':
        TaskConfig(GrinstajnHFDataSource, MAGIC_TELESCOPE_FEATURES),
    'MiniBooNE':
        TaskConfig(GrinstajnHFDataSource, MINI_BOONE_FEATURES),
    'road-safety':
        TaskConfig(GrinstajnHFDataSource, ROAD_SAFETY_FEATURES),
    'pol':
        TaskConfig(GrinstajnHFDataSource, POL_FEATURES),
    'jannis':
        TaskConfig(GrinstajnHFDataSource, JANNIS_FEATURES),
    'house_16H':
        TaskConfig(GrinstajnHFDataSource, HOUSE_16H_FEATURES),
    'amazon':
        TaskConfig(AmazonDataSource, AMAZON_FEATURES),
    'appetency':
        TaskConfig(KddCup2009DataSource, APPETENCY_FEATURES),
    'churn':
        TaskConfig(KddCup2009DataSource, CHURN_FEATURES),
    'upselling':
        TaskConfig(KddCup2009DataSource, UPSELLING_FEATURES),
    'click':
        TaskConfig(ClickDataSource, CLICK_FEATURES),
    'kick':
        TaskConfig(KickDataSource, KICK_FEATURES),
    'product_sentiment_machine_hack':
        TaskConfig(AutoMLBenchmarkDataSource, PROD_FEATURES),
    'data_scientist_salary':
        TaskConfig(AutoMLBenchmarkDataSource, SALARY_FEATURES),
    'melbourne_airbnb':
        TaskConfig(AutoMLBenchmarkDataSource, AIRBNB_FEATURES),
    'news_channel':
        TaskConfig(AutoMLBenchmarkDataSource, NEWS_CHANNEL_FEATURES),
    'wine_reviews':
        TaskConfig(AutoMLBenchmarkDataSource, WINE_REVIEWS_FEATURES),
    'imdb_genre_prediction':
        TaskConfig(AutoMLBenchmarkDataSource, IMDB_FEATURES),
    'fake_job_postings2':
        TaskConfig(AutoMLBenchmarkDataSource, FAKE_JOBS_FEATURES),
    'kick_starter_funding':
        TaskConfig(AutoMLBenchmarkDataSource, KICKSTARTER_FEATURES),
    'jigsaw_unintended_bias100K':
        TaskConfig(AutoMLBenchmarkDataSource, JIGSAW_FEATURES),
    'iris':
        TaskConfig(IrisDataSource, IRIS_FEATURES),
    'dry-bean':
        TaskConfig(DryBeanDataSource, DRY_BEAN_FEATURES),
    'heart-disease':
        TaskConfig(HeartDiseaseDataSource, HEART_DISEASE_FEATURES),
    'wine':
        TaskConfig(WineCultivarsDataSource, WINE_CULTIVARS_FEATURES),
    'wine-quality':
        TaskConfig(WineQualityDataSource, WINE_QUALITY_FEATURES),
    'rice':
        TaskConfig(RiceDataSource, RICE_FEATURES),
    'breast-cancer':
        TaskConfig(BreastCancerDataSource, BREAST_CANCER_FEATURES),
    'cars':
        TaskConfig(CarDataSource, CAR_FEATURES),
    'raisin':
        TaskConfig(RaisinDataSource, RAISIN_FEATURES),
    'abalone':
        TaskConfig(AbaloneDataSource, ABALONE_FEATURES),
    'otto-products':
        TaskConfig(OttoProductsDataSource, OTTO_FEATURES),
    'sf-crime':
        TaskConfig(SfCrimeDataSource, SF_CRIME_FEATURES),
    'plasticc':
        TaskConfig(PlasticcDataSource, PLASTICC_FEATURES),
    'walmart':
        TaskConfig(WalmartDataSource, WALMART_FEATURES),
    'tradeshift':
        TaskConfig(TradeShiftDataSource, TRADESHIFT_FEATURES),
    'schizophrenia':
        TaskConfig(SchizophreniaDataSource, SCHIZOPHRENIA_FEATURES),
    'titanic':
        TaskConfig(TitanicDataSource, TITANIC_FEATURES),
    'santander-transactions':
        TaskConfig(SantanderTransactionDataSource,
                   SANTANDER_TRANSACTION_FEATURES),
    'home-credit-default-risk':
        TaskConfig(HomeCreditDefaultDataSource, HOME_CREDIT_DEFAULT_FEATURES),
    'ieee-fraud-detection':
        TaskConfig(IeeFraudDetectionDataSource, IEEE_FRAUD_DETECTION_FEATURES),
    'safe-driver-prediction':
        TaskConfig(SafeDriverPredictionDataSource,
                   SAFE_DRIVER_PREDICTION_FEATURES),
    'santander-customer-satisfaction':
        TaskConfig(SantanderCustomerSatisfactionDataSource,
                   SANTANDER_CUSTOMER_SATISFACTION_FEATURES),
    'amex-default':
        TaskConfig(AmexDefaultDataSource, AMEX_DEFAULT_FEATURES),
    'ad-fraud':
        TaskConfig(AdFraudDataSource, AD_FRAUD_FEATURES),
    'college_scorecard':
        TaskConfig(CollegeScorecardDataSource, COLLEGE_SCORECARD_FEATURES),
}


def get_task_config(name: str) -> TaskConfig:
    if name in _TASK_REGISTRY:
        return _TASK_REGISTRY[name]
    else:
        raise NotImplementedError(f"task {name} not implemented.")

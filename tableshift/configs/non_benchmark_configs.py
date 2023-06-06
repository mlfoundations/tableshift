"""
Experiment configs for the non-TableShift benchmark tasks.
"""

from tableshift.configs.benchmark_configs import \
    _MIMIC_EXTRACT_PASSTHROUGH_COLUMNS
from tableshift.configs.experiment_config import ExperimentConfig
from tableshift.configs.experiment_defaults import DEFAULT_ID_TEST_SIZE, \
    DEFAULT_OOD_VAL_SIZE, DEFAULT_ID_VAL_SIZE, DEFAULT_RANDOM_STATE
from tableshift.core import RandomSplitter, Grouper, PreprocessorConfig, \
    DomainSplitter, FixedSplitter

GRINSTAJN_TEST_SIZE = 0.21

GRINSZTAJN_VAL_SIZE = 0.09
NON_BENCHMARK_CONFIGS = {
    "adult": ExperimentConfig(
        splitter=FixedSplitter(val_size=0.25, random_state=29746),
        grouper=Grouper({"Race": ["White", ], "Sex": ["Male", ]}, drop=False),
        preprocessor_config=PreprocessorConfig(), tabular_dataset_kwargs={}),

    "_debug": ExperimentConfig(
        splitter=DomainSplitter(
            val_size=0.01,
            id_test_size=0.2,
            ood_val_size=0.25,
            random_state=43406,
            domain_split_varname="purpose",
            # Counts by domain are below. We hold out all of the smallest
            # domains to avoid errors with very small domains during dev.
            # A48       9
            # A44      12
            # A410     12
            # A45      22
            # A46      50
            # A49      97
            # A41     103
            # A42     181
            # A40     234
            # A43     280
            domain_split_ood_values=["A44", "A410", "A45", "A46", "A48"]
        ),
        grouper=Grouper({"sex": ['1.0', ], "age_geq_median": ['1.0', ]},
                        drop=False),
        preprocessor_config=PreprocessorConfig(),
        tabular_dataset_kwargs={"name": "german"}),

    "german": ExperimentConfig(
        splitter=RandomSplitter(val_size=0.01, test_size=0.2, random_state=832),
        grouper=Grouper({"sex": ['1.0', ], "age_geq_median": ['1.0', ]},
                        drop=False),
        preprocessor_config=PreprocessorConfig(), tabular_dataset_kwargs={}),

    "mooc": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname="course_id",
                                domain_split_ood_values=[
                                    "HarvardX/CB22x/2013_Spring"]),
        grouper=Grouper({"gender": ["m", ],
                         "LoE_DI": ["Bachelor's", "Master's", "Doctorate"]},
                        drop=False),
        preprocessor_config=PreprocessorConfig(), tabular_dataset_kwargs={}),

    ################### Grinsztajn et al. benchmark datasets ###################

    **{x: ExperimentConfig(
        splitter=RandomSplitter(val_size=GRINSZTAJN_VAL_SIZE,
                                test_size=GRINSTAJN_TEST_SIZE,
                                random_state=DEFAULT_RANDOM_STATE),
        grouper=None,
        preprocessor_config=PreprocessorConfig(),
        tabular_dataset_kwargs={"dataset_name": x}
    ) for x in ("electricity", "bank-marketing", "california",
                "covertype", "credit", 'default-of-credit-card-clients',
                'eye_movements', 'Higgs', 'MagicTelescope', 'MiniBooNE',
                'road-safety', 'pol', 'jannis', 'house_16H')},

    ################### MetaMIMIC datasets #####################################

    "metamimic_alcohol": ExperimentConfig(
        splitter=RandomSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                test_size=DEFAULT_ID_TEST_SIZE),
        grouper=None,
        preprocessor_config=PreprocessorConfig(
            numeric_features="kbins",
            passthrough_columns=["age"]
        ),
        tabular_dataset_kwargs={'name': 'metamimic_alcohol'}),

    'metamimic_anemia': ExperimentConfig(
        splitter=RandomSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                test_size=DEFAULT_ID_TEST_SIZE),
        grouper=None,
        preprocessor_config=PreprocessorConfig(
            numeric_features="kbins",
            passthrough_columns=["age"]
        ),
        tabular_dataset_kwargs={'name': 'metamimic_anemia'}),

    'metamimic_atrial': ExperimentConfig(
        splitter=RandomSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                test_size=DEFAULT_ID_TEST_SIZE),
        grouper=None,
        preprocessor_config=PreprocessorConfig(
            numeric_features="kbins",
            passthrough_columns=["age"]
        ),
        tabular_dataset_kwargs={'name': 'metamimic_atrial'}),

    'metamimic_diabetes': ExperimentConfig(
        splitter=RandomSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                test_size=DEFAULT_ID_TEST_SIZE),
        grouper=None,
        preprocessor_config=PreprocessorConfig(
            numeric_features="kbins",
            passthrough_columns=["age"]
        ),
        tabular_dataset_kwargs={'name': 'metamimic_diabetes'}),

    'metamimic_heart': ExperimentConfig(
        splitter=RandomSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                test_size=DEFAULT_ID_TEST_SIZE),
        grouper=None,
        preprocessor_config=PreprocessorConfig(
            numeric_features="kbins",
            passthrough_columns=["age"]
        ),
        tabular_dataset_kwargs={'name': 'metamimic_heart'}),

    'metamimic_hypertension': ExperimentConfig(
        splitter=RandomSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                test_size=DEFAULT_ID_TEST_SIZE),
        grouper=None,
        preprocessor_config=PreprocessorConfig(
            numeric_features="kbins",
            passthrough_columns=["age"]
        ),
        tabular_dataset_kwargs={'name': 'metamimic_hypertension'}),

    'metamimic_hypotension': ExperimentConfig(
        splitter=RandomSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                test_size=DEFAULT_ID_TEST_SIZE),
        grouper=None,
        preprocessor_config=PreprocessorConfig(
            numeric_features="kbins",
            passthrough_columns=["age"]
        ),
        tabular_dataset_kwargs={'name': 'metamimic_hypotension'}),

    'metamimic_ischematic': ExperimentConfig(
        splitter=RandomSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                test_size=DEFAULT_ID_TEST_SIZE),
        grouper=None,
        preprocessor_config=PreprocessorConfig(
            numeric_features="kbins",
            passthrough_columns=["age"]
        ),
        tabular_dataset_kwargs={'name': 'metamimic_ischematic'}),

    'metamimic_lipoid': ExperimentConfig(
        splitter=RandomSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                test_size=DEFAULT_ID_TEST_SIZE),
        grouper=None,
        preprocessor_config=PreprocessorConfig(
            numeric_features="kbins",
            passthrough_columns=["age"]
        ),
        tabular_dataset_kwargs={'name': 'metamimic_lipoid'}),

    'metamimic_overweight': ExperimentConfig(
        splitter=RandomSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                test_size=DEFAULT_ID_TEST_SIZE),
        grouper=None,
        preprocessor_config=PreprocessorConfig(
            numeric_features="kbins",
            passthrough_columns=["age"]
        ),
        tabular_dataset_kwargs={'name': 'metamimic_overweight'}),

    'metamimic_purpura': ExperimentConfig(
        splitter=RandomSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                test_size=DEFAULT_ID_TEST_SIZE),
        grouper=None,
        preprocessor_config=PreprocessorConfig(
            numeric_features="kbins",
            passthrough_columns=["age"]
        ),
        tabular_dataset_kwargs={'name': 'metamimic_purpura'}),

    'metamimic_respiratory': ExperimentConfig(
        splitter=RandomSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                test_size=DEFAULT_ID_TEST_SIZE),
        grouper=None,
        preprocessor_config=PreprocessorConfig(
            numeric_features="kbins",
            passthrough_columns=["age"]
        ),
        tabular_dataset_kwargs={'name': 'metamimic_respiratory'}),

    ################### CatBoost benchmark datasets ########################

    "amazon": ExperimentConfig(
        splitter=RandomSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                test_size=DEFAULT_ID_TEST_SIZE,
                                random_state=DEFAULT_RANDOM_STATE),
        grouper=None,
        preprocessor_config=PreprocessorConfig(),
        tabular_dataset_kwargs={
            "kaggle_dataset_name": "amazon-employee-access-challenge"}),

    **{k: ExperimentConfig(
        splitter=RandomSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                test_size=DEFAULT_ID_TEST_SIZE,
                                random_state=DEFAULT_RANDOM_STATE),
        grouper=None,
        # categorical features in this dataset have *extremely* high cardinality
        preprocessor_config=PreprocessorConfig(
            categorical_features="passthrough"),
        tabular_dataset_kwargs={"task_name": k}) for k in
        ("appetency", "churn", "upselling")},

    "click": ExperimentConfig(
        splitter=FixedSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                               random_state=DEFAULT_RANDOM_STATE),
        grouper=None,
        # categorical features in this dataset have *extremely* high cardinality
        preprocessor_config=PreprocessorConfig(
            categorical_features="passthrough"),
        tabular_dataset_kwargs={
            "kaggle_dataset_name": "kddcup2012-track2"}),

    'kick': ExperimentConfig(
        splitter=RandomSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                test_size=DEFAULT_ID_TEST_SIZE,
                                random_state=DEFAULT_RANDOM_STATE),
        grouper=None,
        preprocessor_config=PreprocessorConfig(
            categorical_features="passthrough"),
        tabular_dataset_kwargs={"kaggle_dataset_name": "DontGetKicked"},
    ),

    ############# AutoML benchmark datasets (classification only) ##############
    **{x: ExperimentConfig(
        splitter=FixedSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                               random_state=DEFAULT_RANDOM_STATE),
        grouper=None,
        preprocessor_config=PreprocessorConfig(
            categorical_features="passthrough",
            dropna=None),
        tabular_dataset_kwargs={
            "automl_benchmark_dataset_name": x}) for x in (
        'product_sentiment_machine_hack', 'data_scientist_salary',
        'melbourne_airbnb', 'news_channel', 'wine_reviews',
        'imdb_genre_prediction', 'fake_job_postings2', 'kick_starter_funding',
        'jigsaw_unintended_bias100K',)},

    ######################## UCI datasets ######################################
    **{x: ExperimentConfig(
        splitter=RandomSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                test_size=DEFAULT_ID_TEST_SIZE,
                                random_state=DEFAULT_RANDOM_STATE),
        grouper=None,
        preprocessor_config=PreprocessorConfig(),
        tabular_dataset_kwargs={}
    ) for x in ('iris', 'dry-bean', 'heart-disease', 'wine', 'wine-quality',
                'rice', 'cars', 'raisin', 'abalone')},

    # For breast cancer, mean/stc/worst values are already computed as features,
    # so we passthrough by default.
    'breast-cancer': ExperimentConfig(
        splitter=RandomSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                test_size=DEFAULT_ID_TEST_SIZE,
                                random_state=DEFAULT_RANDOM_STATE),
        grouper=None,
        preprocessor_config=PreprocessorConfig(numeric_features="passthrough"),
        tabular_dataset_kwargs={}),
    ######################## Kaggle datasets ###################################
    **{x: ExperimentConfig(
        splitter=RandomSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                test_size=DEFAULT_ID_TEST_SIZE,
                                random_state=DEFAULT_RANDOM_STATE),
        grouper=None,
        preprocessor_config=PreprocessorConfig(
            categorical_features="passthrough", dropna=None),
        tabular_dataset_kwargs={}
    ) for x in ('otto-products', 'sf-crime', 'plasticc', 'walmart',
                'tradeshift', 'schizophrenia', 'titanic',
                'santander-transactions', 'home-credit-default-risk',
                'ieee-fraud-detection', 'safe-driver-prediction',
                'santander-customer-satisfaction', 'amex-default',
                'ad-fraud')},

    ############################################################################

    "mimic_extract_los_3_selected": ExperimentConfig(
        splitter=RandomSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                test_size=DEFAULT_ID_TEST_SIZE,
                                random_state=DEFAULT_RANDOM_STATE),
        grouper=None,
        preprocessor_config=PreprocessorConfig(
            passthrough_columns=_MIMIC_EXTRACT_PASSTHROUGH_COLUMNS),
        tabular_dataset_kwargs={"task": "los_3",
                                "name": "mimic_extract_los_3_selected"}),

    "mimic_extract_mort_hosp_selected": ExperimentConfig(
        splitter=RandomSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                test_size=DEFAULT_ID_TEST_SIZE,
                                random_state=DEFAULT_RANDOM_STATE),
        grouper=None,
        preprocessor_config=PreprocessorConfig(
            passthrough_columns=_MIMIC_EXTRACT_PASSTHROUGH_COLUMNS),
        tabular_dataset_kwargs={"task": "mort_hosp",
                                "name": "mimic_extract_mort_hosp_selected"}),

    "communities_and_crime": ExperimentConfig(
        splitter=RandomSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                test_size=DEFAULT_ID_TEST_SIZE,
                                random_state=DEFAULT_RANDOM_STATE),
        grouper=None,
        preprocessor_config=PreprocessorConfig(), tabular_dataset_kwargs={}),

    "compas": ExperimentConfig(
        splitter=RandomSplitter(test_size=0.2, val_size=0.01,
                                random_state=90127),
        grouper=Grouper({"race": ["Caucasian", ], "sex": ["Male", ]},
                        drop=False),
        preprocessor_config=PreprocessorConfig(), tabular_dataset_kwargs={}),

}

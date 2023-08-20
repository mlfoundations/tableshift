"""
Experiment configs for the 'official' TableShift benchmark tasks.

All other configs are in non_benchmark_configs.py.
"""

from tableshift.configs.experiment_config import ExperimentConfig
from tableshift.configs.experiment_defaults import DEFAULT_ID_TEST_SIZE, \
    DEFAULT_OOD_VAL_SIZE, DEFAULT_ID_VAL_SIZE, DEFAULT_RANDOM_STATE
from tableshift.core import Grouper, PreprocessorConfig, DomainSplitter
from tableshift.datasets import BRFSS_YEARS, ACS_YEARS, NHANES_YEARS
from tableshift.datasets.mimic_extract import MIMIC_EXTRACT_STATIC_FEATURES
from tableshift.datasets.mimic_extract_feature_lists import \
    MIMIC_EXTRACT_SHARED_FEATURES

# We passthrough all non-static columns because we use
# MIMIC-extract's default preprocessing/imputation and do not
# wish to modify it for these features
# (static features are not preprocessed by MIMIC-extract). See
# tableshift.datasets.mimic_extract.preprocess_mimic_extract().
_MIMIC_EXTRACT_PASSTHROUGH_COLUMNS = [
    f for f in MIMIC_EXTRACT_SHARED_FEATURES.names
    if f not in MIMIC_EXTRACT_STATIC_FEATURES.names]

BENCHMARK_CONFIGS = {
    "acsfoodstamps": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname="DIVISION",
                                domain_split_ood_values=['06']),
        grouper=Grouper({"RAC1P": [1, ], "SEX": [1, ]}, drop=False),
        preprocessor_config=PreprocessorConfig(),
        tabular_dataset_kwargs={"acs_task": "acsfoodstamps"}),

    "acsincome": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname="DIVISION",
                                domain_split_ood_values=['01']),
        grouper=Grouper({"RAC1P": [1, ], "SEX": [1, ]}, drop=False),
        preprocessor_config=PreprocessorConfig(),
        tabular_dataset_kwargs={"acs_task": "acsincome"}),

    "acspubcov": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname="DIS",
                                domain_split_ood_values=['1.0']),
        grouper=Grouper({"RAC1P": [1, ], "SEX": [1, ]}, drop=False),
        preprocessor_config=PreprocessorConfig(),
        tabular_dataset_kwargs={"acs_task": "acspubcov", "name": "acspubcov",
                                "years": ACS_YEARS}),

    "acsunemployment": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname='SCHL',
                                # No high school diploma vs. GED/diploma or higher.
                                domain_split_ood_values=['01', '02', '03', '04',
                                                         '05', '06', '07', '08',
                                                         '09', '10', '11', '12',
                                                         '13', '14', '15']),
        grouper=Grouper({"RAC1P": [1, ], "SEX": [1, ]}, drop=False),
        preprocessor_config=PreprocessorConfig(),
        tabular_dataset_kwargs={"acs_task": "acsunemployment"}),

    # ANES, Split by region; OOD is south: (AL, AR, DE, D.C., FL, GA, KY, LA,
    # MD, MS, NC, OK, SC,TN, TX, VA, WV)
    "anes": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname='VCF0112',
                                domain_split_ood_values=['3.0']),
        # male vs. all others; white non-hispanic vs. others
        grouper=Grouper({"VCF0104": ["1", ], "VCF0105a": ["1.0", ]},
                        drop=False),
        preprocessor_config=PreprocessorConfig(numeric_features="kbins",
                                               dropna=None),
        tabular_dataset_kwargs={}),

    "brfss_blood_pressure": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname="BMI5CAT",
                                # OOD values: [1 underweight, 2 normal weight], [3 overweight, 4 obese]
                                domain_split_ood_values=['3.0', '4.0']),
        grouper=Grouper({"PRACE1": [1, ], "SEX": [1, ]}, drop=False),
        preprocessor_config=PreprocessorConfig(passthrough_columns=["IYEAR"]),
        tabular_dataset_kwargs={"name": "brfss_blood_pressure",
                                "task": "blood_pressure",
                                "years": BRFSS_YEARS},
    ),

    # "White nonhispanic" (in-domain) vs. all other race/ethnicity codes (OOD)
    "brfss_diabetes": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname="PRACE1",
                                domain_split_ood_values=[2, 3, 4, 5, 6],
                                domain_split_id_values=[1, ]),
        grouper=Grouper({"SEX": [1, ]}, drop=False),
        preprocessor_config=PreprocessorConfig(passthrough_columns=["IYEAR"]),
        tabular_dataset_kwargs={"name": "brfss_diabetes",
                                "task": "diabetes", "years": BRFSS_YEARS},
    ),

    "diabetes_readmission": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname='admission_source_id',
                                domain_split_ood_values=[7, ]),
        # male vs. all others; white non-hispanic vs. others
        grouper=Grouper({"race": ["Caucasian", ], "gender": ["Male", ]},
                        drop=False),
        # Note: using min_frequency=0.01 reduces data
        # dimensionality from ~2400 -> 169 columns.
        # This is due to high cardinality of 'diag_*' features.
        preprocessor_config=PreprocessorConfig(min_frequency=0.01),
        tabular_dataset_kwargs={}),

    "heloc": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname='ExternalRiskEstimateLow',
                                domain_split_ood_values=[0]),
        grouper=None,
        preprocessor_config=PreprocessorConfig(),
        tabular_dataset_kwargs={"name": "heloc"},
    ),

    "mimic_extract_los_3": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname="insurance",
                                domain_split_ood_values=["Medicare"]),

        grouper=Grouper({"gender": ['M'], }, drop=False),
        preprocessor_config=PreprocessorConfig(
            passthrough_columns=_MIMIC_EXTRACT_PASSTHROUGH_COLUMNS),
        tabular_dataset_kwargs={"task": "los_3",
                                "name": "mimic_extract_los_3"}),

    "mimic_extract_mort_hosp": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname="insurance",
                                domain_split_ood_values=["Medicare",
                                                         "Medicaid"]),
        grouper=Grouper({"gender": ['M'], }, drop=False),
        preprocessor_config=PreprocessorConfig(
            passthrough_columns=_MIMIC_EXTRACT_PASSTHROUGH_COLUMNS),
        tabular_dataset_kwargs={"task": "mort_hosp",
                                "name": "mimic_extract_mort_hosp"}),

    "nhanes_cholesterol": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname='RIDRETH_merged',
                                domain_split_ood_values=[1, 2, 4, 6, 7],
                                domain_split_id_values=[3],
                                ),
        # Group by male vs. all others
        grouper=Grouper({"RIAGENDR": ["1.0", ]}, drop=False),
        preprocessor_config=PreprocessorConfig(
            passthrough_columns=["nhanes_year"],
            numeric_features="kbins"),
        tabular_dataset_kwargs={"nhanes_task": "cholesterol",
                                "years": NHANES_YEARS}),

    "assistments": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname='school_id',
                                domain_split_ood_values=[5040.0,
                                                         11502.0,
                                                         11318.0,
                                                         11976.0,
                                                         12421.0,
                                                         12379.0,
                                                         11791.0,
                                                         8359.0,
                                                         12406.0,
                                                         7594.0]),
        grouper=None,
        preprocessor_config=PreprocessorConfig(
            passthrough_columns=["skill_id", "bottom_hint", "first_action"],
        ),
        tabular_dataset_kwargs={},
    ),

    "college_scorecard": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname='CCBASIC',
                                domain_split_ood_values=[
                                    'Special Focus Institutions--Other special-focus institutions',
                                    'Special Focus Institutions--Theological seminaries, Bible colleges, and other faith-related institutions',
                                    "Associate's--Private For-profit 4-year Primarily Associate's",
                                    'Baccalaureate Colleges--Diverse Fields',
                                    'Special Focus Institutions--Schools of art, music, and design',
                                    "Associate's--Private Not-for-profit",
                                    "Baccalaureate/Associate's Colleges",
                                    "Master's Colleges and Universities (larger programs)"]
                                ),
        grouper=None,
        preprocessor_config=PreprocessorConfig(
            # Several categorical features in college scorecard have > 10k
            # unique values; so we label-encode instead of one-hot encoding.
            categorical_features="label_encode",
            # Some important numeric features are not reported by universities
            # in a way that could be systematic (and we would like these included
            # in the sample, not excluded), so we use kbins
            numeric_features="kbins",
            n_bins=100,
            dropna=None,
        ),
        tabular_dataset_kwargs={},
    ),

    "nhanes_lead": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname='INDFMPIRBelowCutoff',
                                domain_split_ood_values=[1.]),
        # Race (non. hispanic white vs. all others; male vs. all others)
        grouper=Grouper({"RIDRETH_merged": [3, ], "RIAGENDR": ["1.0", ]},
                        drop=False),
        preprocessor_config=PreprocessorConfig(
            passthrough_columns=["nhanes_year"],
            numeric_features="kbins"),
        tabular_dataset_kwargs={"nhanes_task": "lead", "years": NHANES_YEARS}),

    # LOS >= 47 is roughly the 80th %ile of data.
    "physionet": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname='ICULOS',
                                domain_split_gt_thresh=47.0),
        grouper=None,
        preprocessor_config=PreprocessorConfig(numeric_features="kbins",
                                               dropna=None),
        tabular_dataset_kwargs={"name": "physionet"}),
}

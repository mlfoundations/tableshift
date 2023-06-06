from dataclasses import dataclass
from typing import Sequence, Optional, Any, Iterator

from tableshift.configs.benchmark_configs import ExperimentConfig
from tableshift.configs.experiment_defaults import DEFAULT_ID_TEST_SIZE, \
    DEFAULT_OOD_VAL_SIZE, DEFAULT_ID_VAL_SIZE, DEFAULT_RANDOM_STATE
from tableshift.core import Grouper, PreprocessorConfig, DomainSplitter
from tableshift.datasets import ACS_REGIONS, ACS_YEARS, \
    BRFSS_YEARS, CANDC_STATE_LIST, NHANES_YEARS, ANES_YEARS, \
    ANES_REGIONS, MIMIC_EXTRACT_SHARED_FEATURES, MIMIC_EXTRACT_STATIC_FEATURES
from tableshift.datasets.assistments import SCHOOL_IDS


def _to_nested(ary: Sequence[Any]) -> Sequence[Sequence[Any]]:
    """Create a nested tuple from a sequence.

    This reformats lists e.g. where each element in the list is the only desired
    out-of-domain value in an experiment.
    """
    return tuple([x] for x in ary)


@dataclass
class DomainShiftExperimentConfig:
    """Class to hold parameters for a domain shift experiment.

    This class defines a *set* of experiments, where the distribution split changes
    over experiments but all other factors (preprocessing, grouping, etc.) stay fixed.

    This class is used e.g. to identify which of a set of candidate domain splits has the
    biggest domain gap.
    """
    tabular_dataset_kwargs: dict
    domain_split_varname: str
    domain_split_ood_values: Sequence[Any]
    grouper: Optional[Grouper]
    preprocessor_config: PreprocessorConfig
    domain_split_id_values: Optional[Sequence[Any]] = None

    def as_experiment_config_iterator(
            self, val_size=DEFAULT_ID_VAL_SIZE,
            ood_val_size=DEFAULT_OOD_VAL_SIZE,
            id_test_size=DEFAULT_ID_TEST_SIZE,
            random_state=DEFAULT_RANDOM_STATE
    ) -> Iterator[ExperimentConfig]:
        for i, tgt in enumerate(self.domain_split_ood_values):
            if self.domain_split_id_values is not None:
                src = self.domain_split_id_values[i]
            else:
                src = None
            if not isinstance(tgt, tuple) and not isinstance(tgt, list):
                tgt = (tgt,)
            splitter = DomainSplitter(
                val_size=val_size,
                ood_val_size=ood_val_size,
                id_test_size=id_test_size,
                domain_split_varname=self.domain_split_varname,
                domain_split_ood_values=tgt,
                domain_split_id_values=src,
                random_state=random_state)
            yield ExperimentConfig(splitter=splitter, grouper=self.grouper,
                                   preprocessor_config=self.preprocessor_config,
                                   tabular_dataset_kwargs=self.tabular_dataset_kwargs)


# Set of fixed domain shift experiments.
domain_shift_experiment_configs = {
    "acsfoodstamps_region": DomainShiftExperimentConfig(
        tabular_dataset_kwargs={"name": "acsfoodstamps",
                                "acs_task": "acsfoodstamps"},
        domain_split_varname="DIVISION",
        domain_split_ood_values=_to_nested(ACS_REGIONS),
        grouper=Grouper({"RAC1P": [1, ], "SEX": [1, ]}, drop=False),
        preprocessor_config=PreprocessorConfig()),

    "acsincome_region": DomainShiftExperimentConfig(
        tabular_dataset_kwargs={"name": "acsincome",
                                "acs_task": "acsincome"},
        domain_split_varname="DIVISION",
        domain_split_ood_values=_to_nested(ACS_REGIONS),
        grouper=Grouper({"RAC1P": [1, ], "SEX": [1, ]}, drop=False),
        preprocessor_config=PreprocessorConfig()),

    "acspubcov_disability": DomainShiftExperimentConfig(
        tabular_dataset_kwargs={"name": "acspubcov",
                                "acs_task": "acspubcov",
                                "years": ACS_YEARS},
        domain_split_varname="DIS",
        domain_split_ood_values=_to_nested(['1.0']),
        grouper=Grouper({"RAC1P": [1, ], "SEX": [1, ]}, drop=False),
        preprocessor_config=PreprocessorConfig()),

    "acsunemployment_edlvl": DomainShiftExperimentConfig(
        tabular_dataset_kwargs={"name": "acsunemployment",
                                "acs_task": "acsunemployment"},
        domain_split_varname='SCHL',
        # No high school diploma vs. GED/diploma or higher.
        domain_split_ood_values=[['01', '02', '03', '04',
                                  '05', '06', '07', '08',
                                  '09', '10', '11', '12',
                                  '13', '14', '15'],
                                 ['16', '17', '18', '19',
                                  '20', '21', '22', '23', '24']],
        grouper=Grouper({"RAC1P": [1, ], "SEX": [1, ]}, drop=False),
        preprocessor_config=PreprocessorConfig()),

    "acsunemployment_mobility": DomainShiftExperimentConfig(
        tabular_dataset_kwargs={"name": "acsunemployment",
                                "acs_task": "acsunemployment"},
        domain_split_varname='MIG',
        domain_split_ood_values=[['02', '03']],
        grouper=Grouper({"RAC1P": [1, ], "SEX": [1, ]}, drop=False),
        preprocessor_config=PreprocessorConfig()),

    "assistments": DomainShiftExperimentConfig(
        tabular_dataset_kwargs={"name": "assistments"},
        domain_split_varname='school_id',
        domain_split_ood_values=SCHOOL_IDS,
        grouper=None,
        preprocessor_config=PreprocessorConfig(
            passthrough_columns=["skill_id", "bottom_hint", "first_action"],
        ),
    ),

    "college_scorecard_ccbasic": DomainShiftExperimentConfig(
        tabular_dataset_kwargs={"name": "college_scorecard"},
        domain_split_varname='CCBASIC',
        domain_split_ood_values=[
            "Associate's--Public Urban-serving Multicampus",
            "Associate's--Public 2-year colleges under 4-year universities",
            "Associate's--Public Suburban-serving Single Campus",
            'Special Focus Institutions--Schools of business and management',
            "Associate's--Public Rural-serving Small",
            "Associate's--Public Rural-serving Medium",
            'Baccalaureate Colleges--Arts & Sciences',
            'Special Focus Institutions--Theological seminaries, Bible colleges, and other faith-related institutions',
            'Research Universities (high research activity)',
            "Associate's--Private For-profit",
            "Associate's--Public Rural-serving Large",
            "Master's Colleges and Universities (larger programs)",
            "Baccalaureate/Associate's Colleges",
            "Master's Colleges and Universities (smaller programs)",
            'Tribal Colleges',
            'Special Focus Institutions--Schools of art, music, and design',
            "Associate's--Private Not-for-profit",
            "Associate's--Public Suburban-serving Multicampus",
            "Associate's--Private For-profit 4-year Primarily Associate's",
            "Master's Colleges and Universities (medium programs)",
            "Associate's--Public 4-year Primarily Associate's",
            "Associate's--Public Urban-serving Single Campus",
            'Special Focus Institutions--Schools of law',
            'Baccalaureate Colleges--Diverse Fields',
            'Research Universities (very high research activity)',
            'Special Focus Institutions--Other health professions schools',
            'Doctoral/Research Universities',
            'Special Focus Institutions--Other technology-related schools',
            'Special Focus Institutions--Other special-focus institutions',
            'Special Focus Institutions--Medical schools and medical centers',
            'Special Focus Institutions--Schools of engineering',
            "Associate's--Private Not-for-profit 4-year Primarily Associate's",
            "Associate's--Public Special Use"],
        grouper=None,
        preprocessor_config=PreprocessorConfig(
            categorical_features="label_encode",
            numeric_features="kbins",
            n_bins=100,
            dropna=None,
        ),

    ),

    "college_scorecard_ccsizset": DomainShiftExperimentConfig(
        tabular_dataset_kwargs={"name": "college_scorecard"},
        domain_split_varname='CCSIZSET',
        domain_split_ood_values=[
            'Small 2-year (confers associate’s degrees, FTE enrollment 500 to 1,999)',
            'Medium 2-year (confers associate’s degrees, FTE enrollment 2000 to 4,999)',
            'Large 2-year (confers associate’s degrees, FTE enrollment 5000 to 9,999)',
            'Small 4-year, highly residential (confers bachelor’s degrees, FTE enrollment 1,000 to 2,999, at least 50 percent of degree-seeking undergraduates live on campus and at least 80 percent attend full time)',
            'Medium 4-year, highly residential (confers bachelor’s degrees, FTE enrollment 3,000 to 9,999, at least 50 percent of degree-seeking undergraduates live on campus and at least 80 percent attend full time)',
            'Very large 2-year (confers associate’s degrees, FTE enrollment 10,000 or more)',
            'Very small 4-year, primarily residential (confers bachelor’s degrees, FTE enrollment less than 1,000, 25 to 49 percent of degree-seeking undergraduates live on campus and at least 50 percent attend full time)',
            'Very small 2-year (confers associate’s degrees, FTE enrollment less than 500)',
            'Small 4-year, primarily nonresidential (confers bachelor’s degrees, FTE enrollment 1,000 to 2,999, less than 25 percent of degree-seeking undergraduates live on campus and/or less than 50 percent attend full time)',
            'Large 4-year, primarily residential (confers bachelor’s degrees, FTE enrollment over 9,999, 25 to 49 percent of degree-seeking undergraduates live on campus and at least 50 percent attend full time)',
            'Medium 4-year, primarily residential (confers bachelor’s degrees, FTE enrollment 3,000 to 9,999, 25 to 49 percent of degree-seeking undergraduates live on campus and at least 50 percent attend full time)',
            'Large 4-year, primarily nonresidential (confers bachelor’s degrees, FTE enrollment over 9,999, less than 25 percent of degree-seeking undergraduates live on campus and/or less than 50 percent attend full time)',
            'Very small 4-year, highly residential (confers bachelor’s degrees, FTE enrollment less than 1,000, at least 50 percent of degree-seeking undergraduates live on campus and at least 80 percent attend full time)',
            'Very small 4-year, primarily nonresidential (confers bachelor’s degrees, FTE enrollment less than 1,000, less than 25 percent of degree-seeking undergraduates live on campus and/or less than 50 percent attend full time)',
            'Small 4-year, primarily residential (confers bachelor’s degrees, FTE enrollment 1,000 to 2,999, 25 to 49 percent of degree-seeking undergraduates live on campus and at least 50 percent attend full time)',
            'Medium 4-year, primarily nonresidential (confers bachelor’s degrees, FTE enrollment 3,000 to 9,999, less than 25 percent of degree-seeking undergraduates live on campus and/or less than 50 percent attend full time)',
            'Large 4-year, highly residential (confers bachelor’s degrees, FTE enrollment over 9,999, at least 50 percent of degree-seeking undergraduates live on campus and at least 80 percent attend full time)',
            'Not applicable, special-focus institution'],
        grouper=None,
        preprocessor_config=PreprocessorConfig(
            categorical_features="label_encode",
            numeric_features="kbins",
            n_bins=100,
            dropna=None,
        ),

    ),

    "brfss_diabetes_race": DomainShiftExperimentConfig(
        tabular_dataset_kwargs={"name": "brfss_diabetes", "task": "diabetes",
                                "years": BRFSS_YEARS},
        domain_split_varname="PRACE1",
        # Train on white nonhispanic; test on all other race identities.
        domain_split_ood_values=[[2, 3, 4, 5, 6]],
        domain_split_id_values=_to_nested([1, ]),
        grouper=Grouper({"SEX": [1, ]}, drop=False),
        preprocessor_config=PreprocessorConfig(
            passthrough_columns=["IYEAR"]), ),

    "brfss_blood_pressure_income": DomainShiftExperimentConfig(
        tabular_dataset_kwargs={"name": "brfss_blood_pressure",
                                "task": "blood_pressure",
                                "years": (2019, 2021)},
        domain_split_varname="POVERTY",
        # Train on non-poverty observations; test (OOD) on poverty observations
        domain_split_ood_values=_to_nested([1, ]),
        domain_split_id_values=_to_nested([0, ]),
        grouper=Grouper({"SEX": [1, ]}, drop=False),
        preprocessor_config=PreprocessorConfig(
            passthrough_columns=["IYEAR"]), ),

    "brfss_blood_pressure_bmi": DomainShiftExperimentConfig(
        tabular_dataset_kwargs={"name": "brfss_blood_pressure",
                                "task": "blood_pressure",
                                "years": BRFSS_YEARS},
        domain_split_varname="BMI5CAT",
        # OOD values: [1 underweight, 2 normal weight], [3 overweight, 4 obese]
        domain_split_ood_values=[['1.0', '2.0'], ['3.0', '4.0']],
        grouper=Grouper({"SEX": [1, ]}, drop=False),
        preprocessor_config=PreprocessorConfig(
            passthrough_columns=["IYEAR"]), ),

    "candc_st": DomainShiftExperimentConfig(
        tabular_dataset_kwargs={"name": "communities_and_crime"},
        domain_split_varname="state",
        domain_split_ood_values=_to_nested(CANDC_STATE_LIST),
        grouper=Grouper({"Race": [1, ], "income_level_above_median": [1, ]},
                        drop=False),
        preprocessor_config=PreprocessorConfig(),
    ),

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
    "_debug": DomainShiftExperimentConfig(
        tabular_dataset_kwargs={"name": "german"},
        domain_split_varname="purpose",
        domain_split_ood_values=[["A44", "A410", "A45", "A46", "A48"]],
        grouper=Grouper({"sex": ['1', ], "age_geq_median": ['1', ]},
                        drop=False),
        preprocessor_config=PreprocessorConfig(),
    ),

    # Integer identifier corresponding to 21 distinct values, for example, physician referral,
    # emergency room, and transfer from a hospital,
    # https://downloads.hindawi.com/journals/bmri/2014/781670.pdf
    "diabetes_admsrc": DomainShiftExperimentConfig(
        tabular_dataset_kwargs={"name": "diabetes_readmission"},
        domain_split_varname='admission_source_id',
        domain_split_ood_values=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 17,
                                 20, 22, 25],
        grouper=Grouper({"race": ["Caucasian", ], "gender": ["Male", ]},
                        drop=False),
        preprocessor_config=PreprocessorConfig(min_frequency=0.01),
    ),

    "heloc_externalrisk": DomainShiftExperimentConfig(
        tabular_dataset_kwargs={"name": "heloc"},
        domain_split_varname='ExternalRiskEstimateLow',
        domain_split_ood_values=[[0], [1]],
        grouper=None,
        preprocessor_config=PreprocessorConfig(),
    ),

    "mooc_course": DomainShiftExperimentConfig(
        tabular_dataset_kwargs={"name": "mooc"},
        domain_split_varname="course_id",
        domain_split_ood_values=['HarvardX/CB22x/2013_Spring',
                                 'HarvardX/CS50x/2012',
                                 'HarvardX/ER22x/2013_Spring',
                                 'HarvardX/PH207x/2012_Fall',
                                 'HarvardX/PH278x/2013_Spring'],
        grouper=Grouper({"gender": ["m", ],
                         "LoE_DI": ["Bachelor's", "Master's", "Doctorate"]},
                        drop=False),
        preprocessor_config=PreprocessorConfig(),
    ),

    "nhanes_cholesterol_race": DomainShiftExperimentConfig(
        tabular_dataset_kwargs={"name": "nhanes_cholesterol",
                                "nhanes_task": "cholesterol",
                                "years": NHANES_YEARS},
        domain_split_varname='RIDRETH_merged',
        domain_split_ood_values=[[1, 2, 4, 6, 7]],
        domain_split_id_values=[[3]],
        # Group by male vs. all others
        grouper=Grouper({"RIAGENDR": ["1.0", ]}, drop=False),
        preprocessor_config=PreprocessorConfig(
            passthrough_columns=["nhanes_year"],
            numeric_features="kbins")
    ),

    "nhanes_lead_poverty": DomainShiftExperimentConfig(
        tabular_dataset_kwargs={"name": "nhanes_lead",
                                "nhanes_task": "lead", "years": NHANES_YEARS},
        domain_split_varname='INDFMPIRBelowCutoff',
        domain_split_ood_values=[[1.]],
        # Race (non. hispanic white vs. all others; male vs. all others)
        grouper=Grouper({"RIDRETH_merged": [3, ], "RIAGENDR": ["1.0", ]},
                        drop=False),
        preprocessor_config=PreprocessorConfig(
            passthrough_columns=["nhanes_year"],
            numeric_features="kbins"),
    ),

    "physionet_set": DomainShiftExperimentConfig(
        tabular_dataset_kwargs={"name": "physionet"},
        domain_split_varname="set",
        domain_split_ood_values=_to_nested(["a", "b"]),
        grouper=Grouper({"Age": [x for x in range(40, 100)], "Gender": [1, ]},
                        drop=False),
        preprocessor_config=PreprocessorConfig(numeric_features="kbins",
                                               dropna=None)
    ),
    "physionet_los": DomainShiftExperimentConfig(
        tabular_dataset_kwargs={"name": "physionet"},
        domain_split_varname="set",
        domain_split_ood_values=_to_nested(["a", "b"]),
        grouper=Grouper({"Age": [x for x in range(40, 100)], "Gender": [1, ]},
                        drop=False),
        preprocessor_config=PreprocessorConfig(numeric_features="kbins",
                                               dropna=None)
    ),

    "anes_region": DomainShiftExperimentConfig(
        tabular_dataset_kwargs={"name": "anes", "years": [2020, ]},
        domain_split_varname='VCF0112',
        domain_split_ood_values=_to_nested(ANES_REGIONS),
        grouper=Grouper({"VCF0104": ["1", ], "VCF0105a": ["1.0", ]},
                        drop=False),
        preprocessor_config=PreprocessorConfig(numeric_features="kbins",
                                               dropna=None)),

    # ANES: test on (2016) or (2020); train on all years prior.
    "anes_year": DomainShiftExperimentConfig(
        tabular_dataset_kwargs={"name": "anes"},
        domain_split_varname="VCF0004",
        domain_split_ood_values=[[ANES_YEARS[-2]], [ANES_YEARS[-1]]],
        domain_split_id_values=[ANES_YEARS[:-2], ANES_YEARS[:-1]],
        grouper=Grouper({"VCF0104": ["1", ], "VCF0105a": ["1.0", ]},
                        drop=False),
        preprocessor_config=PreprocessorConfig(numeric_features="kbins",
                                               dropna=None)),

    "mimic_extract_los_3_ins": DomainShiftExperimentConfig(
        tabular_dataset_kwargs={'task': 'los_3', 'name': 'mimic_extract_los_3'},
        domain_split_varname="insurance",
        domain_split_ood_values=_to_nested(
            ["Medicare", "Medicaid", "Government", "Self Pay"]),
        grouper=Grouper({"gender": ['M'], }, drop=False),
        # We passthrough all non-static columns because we use MIMIC-extract's default
        # preprocessing/imputation and do not wish to modify it for these features
        # (static features are not preprocessed by MIMIC-extract). See
        # tableshift.datasets.mimic_extract.preprocess_mimic_extract().
        preprocessor_config=PreprocessorConfig(
            passthrough_columns=[f for f in MIMIC_EXTRACT_SHARED_FEATURES.names
                                 if
                                 f not in MIMIC_EXTRACT_STATIC_FEATURES.names])),

    "mimic_extract_mort_hosp_ins": DomainShiftExperimentConfig(
        tabular_dataset_kwargs={'task': 'mort_hosp',
                                'name': 'mimic_extract_mort_hosp'},
        domain_split_varname="insurance",
        domain_split_ood_values=_to_nested(
            ["Medicare", "Medicaid", "Government", "Self Pay"]),
        grouper=Grouper({"gender": ['M'], }, drop=False),
        # We passthrough all non-static columns because we use MIMIC-extract's default
        # preprocessing/imputation and do not wish to modify it for these features
        # (static features are not preprocessed by MIMIC-extract). See
        # tableshift.datasets.mimic_extract.preprocess_mimic_extract().
        preprocessor_config=PreprocessorConfig(
            passthrough_columns=[f for f in MIMIC_EXTRACT_SHARED_FEATURES.names
                                 if
                                 f not in MIMIC_EXTRACT_STATIC_FEATURES.names])),
}

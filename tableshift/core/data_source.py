"""Data sources for TableBench."""
import glob
import gzip
import logging
import os
import re
import zipfile
from abc import ABC, abstractmethod
from collections import defaultdict
from functools import partial
from io import StringIO
from typing import Sequence, Optional, Callable

import datasets
import folktables
import numpy as np
import pandas as pd
import requests

import tableshift.datasets
from tableshift.core import utils
from tableshift.datasets.acs import ACS_STATE_LIST, preprocess_acs, \
    get_acs_data_source, ACS_TASK_CONFIGS, acs_data_to_df
from tableshift.datasets.acs_feature_mappings import get_feature_mapping
from tableshift.datasets.adult import ADULT_RESOURCES, ADULT_FEATURE_NAMES, \
    preprocess_adult
from tableshift.datasets.anes import preprocess_anes
from tableshift.datasets.automl_multimodal_benchmark import preprocess_automl
from tableshift.datasets.brfss import preprocess_brfss, align_brfss_features
from tableshift.datasets.catboost_benchmarks import preprocess_appetency, \
    preprocess_click, preprocess_kick
from tableshift.datasets.communities_and_crime import CANDC_RESOURCES, \
    preprocess_candc, CANDC_INPUT_FEATURES
from tableshift.datasets.compas import COMPAS_RESOURCES, preprocess_compas
from tableshift.datasets.diabetes_readmission import \
    DIABETES_READMISSION_RESOURCES, preprocess_diabetes_readmission
from tableshift.datasets.german import GERMAN_RESOURCES, preprocess_german
from tableshift.datasets.grinsztajn import preprocess_grinsztain_datataset
from tableshift.datasets.heloc import preprocess_heloc
from tableshift.datasets.kaggle import preprocess_otto, preprocess_walmart
from tableshift.datasets.mimic_extract import preprocess_mimic_extract, \
    MIMIC_EXTRACT_STATIC_FEATURES
from tableshift.datasets.mooc import preprocess_mooc
from tableshift.datasets.nhanes import preprocess_nhanes_cholesterol, \
    get_nhanes_data_sources, preprocess_nhanes_lead, NHANES_YEARS
from tableshift.datasets.physionet import preprocess_physionet
from tableshift.datasets.uci import WINE_CULTIVARS_FEATURES, ABALONE_FEATURES, \
    preprocess_abalone
from tableshift.datasets.utils import apply_column_missingness_threshold


class DataSource(ABC):
    """Abstract class to represent a generic data source."""

    def __init__(self, cache_dir: str,
                 preprocess_fn: Callable[[pd.DataFrame], pd.DataFrame],
                 resources: Sequence[str] = None,
                 download: bool = True,
                 ):
        self.cache_dir = cache_dir
        self.download = download

        self.preprocess_fn = preprocess_fn
        self.resources = resources
        self._initialize_cache_dir()

    def _initialize_cache_dir(self):
        """Create cache_dir if it does not exist."""
        utils.initialize_dir(self.cache_dir)

    def get_data(self) -> pd.DataFrame:
        """Fetch data from local cache or download if necessary."""
        self._download_if_not_cached()
        raw_data = self._load_data()
        return self.preprocess_fn(raw_data)

    def _download_if_not_cached(self):
        """Download files if they are not already cached."""
        for url in self.resources:
            utils.download_file(url, self.cache_dir)

    @abstractmethod
    def _load_data(self) -> pd.DataFrame:
        """Load the raw data from disk and return it.

        Any preprocessing should be performed in preprocess_fn, not here."""
        raise

    @property
    def is_cached(self) -> bool:
        """Check whether all resources exist in cache dir."""
        for url in self.resources:
            basename = utils.basename_from_url(url)
            fp = os.path.join(self.cache_dir, basename)
            if not os.path.exists(fp):
                return False
        return True


class OfflineDataSource(DataSource):

    def get_data(self) -> pd.DataFrame:
        raw_data = self._load_data()
        return self.preprocess_fn(raw_data)

    def _load_data(self) -> pd.DataFrame:
        raise


class ANESDataSource(OfflineDataSource):
    def __init__(
            self,
            years: Optional[Sequence] = None,
            preprocess_fn=preprocess_anes,
            resources=("anes_timeseries_cdf_csv_20220916/"
                       "anes_timeseries_cdf_csv_20220916.csv",),
            **kwargs):
        if years is not None:
            assert isinstance(years, list) or isinstance(years, tuple), \
                f"years must be a list or tuple, not type {type(years)}."
        self.years = years
        super().__init__(resources=resources,
                         preprocess_fn=preprocess_fn,
                         **kwargs)

    def _load_data(self) -> pd.DataFrame:
        fp = os.path.join(self.cache_dir, self.resources[0])
        df = pd.read_csv(fp, low_memory=False, na_values=(' '))
        if self.years:
            df = df[df["VCF0004"].isin(self.years)]
        return df


class MOOCDataSource(OfflineDataSource):
    def __init__(
            self,
            preprocess_fn=preprocess_mooc,
            resources=(os.path.join("dataverse_files",
                                    "HXPC13_DI_v3_11-13-2019.csv"),),
            **kwargs):
        super().__init__(resources=resources,
                         preprocess_fn=preprocess_fn,
                         **kwargs)

    def _load_data(self) -> pd.DataFrame:
        fp = os.path.join(self.cache_dir, self.resources[0])
        if not os.path.exists(fp):
            raise RuntimeError(
                f"""Data files does not exist at {fp}. This dataset must be 
                manually downloaded. Visit https://doi.org/10.7910/DVN/26147,
                click 'Access Dataset' > 'Original Format ZIP', download the ZIP
                file to the cache directory at {self.cache_dir}, and 
                unzip it.""")
        df = pd.read_csv(fp)
        return df


class KaggleDataSource(DataSource):
    def __init__(
            self,
            kaggle_dataset_name: str,
            kaggle_creds_dir="~/.kaggle",
            **kwargs):
        self.kaggle_creds_dir = kaggle_creds_dir
        self.kaggle_dataset_name = kaggle_dataset_name
        super().__init__(**kwargs)

    @property
    def kaggle_creds(self):
        return os.path.expanduser(
            os.path.join(self.kaggle_creds_dir, "kaggle.json"))

    @property
    def zip_file_name(self):
        """Name of the zip file downloaded by Kaggle API."""
        return os.path.basename(self.kaggle_dataset_name) + ".zip"

    @abstractmethod
    def _load_data(self) -> pd.DataFrame:
        # Should be implemented by subclasses.
        raise

    def _check_creds(self):
        # Check Kaggle authentication.
        assert os.path.exists(self.kaggle_creds), \
            f"No kaggle credentials found at {self.kaggle_creds}."
        "Create an access token at https://www.kaggle.com/YOUR_USERNAME/account"
        f"and store it at {self.kaggle_creds}."

    def _download_kaggle_data(self):
        """Download the data from Kaggle."""
        self._check_creds()

        # Download the dataset using Kaggle CLI.
        cmd = "kaggle datasets download " \
              f"-d {self.kaggle_dataset_name} " \
              f"-p {self.cache_dir}"
        utils.run_in_subprocess(cmd)
        return

    def _download_if_not_cached(self):
        self._download_kaggle_data()
        # location of the local zip file
        zip_fp = os.path.join(self.cache_dir, self.zip_file_name)
        # where to unzip the file to
        unzip_dest = os.path.join(self.cache_dir, self.kaggle_dataset_name)
        with zipfile.ZipFile(zip_fp, 'r') as zf:
            zf.extractall(unzip_dest)


class KaggleDownloadError(ValueError):
    pass


class KaggleCompetitionDataSource(KaggleDataSource):
    def _download_kaggle_data(self):
        """Download the data from Kaggle."""
        self._check_creds()

        # Download using Kaggle CLI.

        cmd = "kaggle competitions download " \
              f"{self.kaggle_dataset_name} " \
              f"-p {self.cache_dir}"
        res = utils.run_in_subprocess(cmd)

        if res.returncode != 0:
            raise KaggleDownloadError(
                f"exception when downloading data for competition "
                f"{self.kaggle_dataset_name} you may"
                "need to visit the competition page on kaggle at "
                f"https://www.kaggle.com/competitions/{self.kaggle_dataset_name}/data"
                " and agree to the terms of the competition.")

        return


class AmazonDataSource(KaggleCompetitionDataSource):
    def __init__(self, preprocess_fn=lambda x: x, **kwargs):
        super().__init__(preprocess_fn=preprocess_fn, **kwargs)

    def _load_data(self) -> pd.DataFrame:
        # only use the training data, since Kaggle set sets are unlabeled.
        train_fp = os.path.join(self.cache_dir, self.kaggle_dataset_name,
                                "train.csv")
        return pd.read_csv(train_fp)


class BRFSSDataSource(DataSource):
    """BRFSS data source.

    Note that the BRFSS is composed of three components: 'fixed core' questions,
    asked every year, 'rotating core', asked every other year, and 'emerging
    core'. Since some of our features come from the rotating core, we only
    use every-other-year data sources; some features would be empty for the
    intervening years.

    See also https://www.cdc.gov/brfss/about/brfss_faq.htm , "What are the
    components of the BRFSS questionnaire?"
    """

    def __init__(self, task: str, preprocess_fn=preprocess_brfss,
                 years=(2021,), **kwargs):
        self.years = years
        resources = tuple([
            f"https://www.cdc.gov/brfss/annual_data/{y}/files/LLCP{y}XPT.zip"
            for y in self.years])
        _preprocess_fn = partial(preprocess_fn, task=task)
        super().__init__(preprocess_fn=_preprocess_fn, resources=resources,
                         **kwargs)

    def _load_data(self) -> pd.DataFrame:
        dfs = {}
        for url in self.resources:
            zip_fname = utils.basename_from_url(url)
            xpt_fname = zip_fname.replace("XPT.zip", ".XPT")
            xpt_fp = os.path.join(self.cache_dir, xpt_fname)
            # Unzip the file if needed
            if not os.path.exists(xpt_fp):
                zip_fp = os.path.join(self.cache_dir, zip_fname)
                logging.debug(f"unzipping {zip_fp}")
                with zipfile.ZipFile(zip_fp, 'r') as zf:
                    zf.extractall(self.cache_dir)
                # BRFSS data files have an awful space at the end; remove it.
                os.rename(xpt_fp + " ", xpt_fp)
            # Read the XPT data
            logging.debug(f"reading {xpt_fp}")
            df = utils.read_xpt(xpt_fp)
            df = align_brfss_features(df)
            dfs[url] = df

        return pd.concat(dfs.values(), axis=0)


class NHANESDataSource(DataSource):
    def __init__(
            self,
            nhanes_task: str,
            years: Sequence[int] = NHANES_YEARS,
            **kwargs):
        self.nhanes_task = nhanes_task
        self.years = years

        if self.nhanes_task == "cholesterol":
            preprocess_fn = preprocess_nhanes_cholesterol
        elif self.nhanes_task == "lead":
            preprocess_fn = preprocess_nhanes_lead
        else:
            raise ValueError

        super().__init__(preprocess_fn=preprocess_fn,
                         **kwargs)

    def _download_if_not_cached(self):

        def _add_suffix_to_fname_from_url(url: str, suffix: str):
            """Helper function to add unique names to files by year."""
            fname = utils.basename_from_url(url)
            f, extension = fname.rsplit(".")
            new_fp = f + suffix + "." + extension
            return new_fp

        sources = get_nhanes_data_sources(self.nhanes_task, self.years)
        for year, urls in sources.items():
            for url in urls:
                destfile = _add_suffix_to_fname_from_url(url, str(year))
                utils.download_file(url, self.cache_dir,
                                    dest_file_name=destfile)

    def _load_data(self) -> pd.DataFrame:
        files = glob.glob(os.path.join(self.cache_dir, "*.XPT"))

        # Initialize a dict where keys are years, and values are lists
        # containing the list of dataframes of data for that year; these
        # can be joined on their index (not sure whether the index is
        # unique across years).
        year_dfs = defaultdict(list)

        for f in files:
            year = int(re.search(".*([0-9]{4})\\.XPT", f).group(1))
            if year in self.years:
                logging.debug(f"reading {f}")
                df = utils.read_xpt(f)
                try:
                    df.set_index("SEQN", inplace=True)
                except KeyError:
                    # Some LLCP files only contain 'SEQNO' feature.
                    df.set_index("SEQNO", inplace=True)
                year_dfs[year].append(df)

        df_list = []
        for year in year_dfs.keys():
            # Join the first dataframe with all others.
            dfs = year_dfs[year]
            src_df = dfs[0]
            try:
                logging.info(f"joining {len(dfs)} dataframes for {year}")
                df = src_df.join(dfs[1:], how="outer")
                df["nhanes_year"] = int(year)
                logging.info("finished joins")
                df_list.append(df)
            except Exception as e:
                logging.error(e)

        if len(df_list) > 1:
            df = pd.concat(df_list, axis=0)
        else:
            df = df_list[0]

        return df


class ACSDataSource(DataSource):
    def __init__(self,
                 acs_task: str,
                 preprocess_fn=preprocess_acs,
                 years: Sequence[int] = (2018,),
                 states=ACS_STATE_LIST,
                 feature_mapping="coarse",
                 **kwargs):
        self.acs_task = acs_task.lower().replace("acs", "")
        self.feature_mapping = get_feature_mapping(feature_mapping)
        self.states = states
        self.years = years
        super().__init__(preprocess_fn=preprocess_fn, **kwargs)

    def _get_acs_data(self):
        year_dfs = []

        for year in self.years:
            logging.info(f"fetching ACS data for year {year}...")
            data_source = get_acs_data_source(year, self.cache_dir)
            year_data = data_source.get_data(states=self.states,
                                             join_household=True,
                                             download=True)
            year_data["ACS_YEAR"] = year
            year_dfs.append(year_data)
        logging.info("fetching ACS data complete.")
        return pd.concat(year_dfs, axis=0)

    def _download_if_not_cached(self):
        """No-op for ACS data; folktables already downloads or uses cache as
        needed at _load_data(). """
        return

    def _load_data(self) -> pd.DataFrame:
        acs_data = self._get_acs_data()
        task_config = ACS_TASK_CONFIGS[self.acs_task]
        target_transform = partial(task_config.target_transform,
                                   threshold=task_config.threshold)
        ACSProblem = folktables.BasicProblem(
            features=task_config.features_to_use.predictors,
            target=task_config.target,
            target_transform=target_transform,
            preprocess=task_config.preprocess,
            postprocess=task_config.postprocess,
        )
        X, y, _ = ACSProblem.df_to_numpy(acs_data)
        df = acs_data_to_df(X, y, task_config.features_to_use,
                            feature_mapping=self.feature_mapping)
        return df


class AdultDataSource(DataSource):
    """Data source for the Adult dataset."""

    def __init__(self, resources=ADULT_RESOURCES,
                 preprocess_fn=preprocess_adult, **kwargs):
        super().__init__(resources=resources,
                         preprocess_fn=preprocess_fn, **kwargs)

    def _load_data(self) -> pd.DataFrame:
        train_fp = os.path.join(self.cache_dir, "adult.data")
        train = pd.read_csv(
            train_fp,
            names=ADULT_FEATURE_NAMES,
            sep=r'\s*,\s*',
            engine='python', na_values="?")
        train["Split"] = "train"

        test_fp = os.path.join(self.cache_dir, "adult.test")

        test = pd.read_csv(
            test_fp,
            names=ADULT_FEATURE_NAMES,
            sep=r'\s*,\s*',
            engine='python', na_values="?", skiprows=1)
        test["Split"] = "test"

        return pd.concat((train, test))


class COMPASDataSource(DataSource):
    def __init__(self, resources=COMPAS_RESOURCES,
                 preprocess_fn=preprocess_compas, **kwargs):
        super().__init__(resources=resources,
                         preprocess_fn=preprocess_fn, **kwargs)

    def _load_data(self) -> pd.DataFrame:
        df = pd.read_csv(
            os.path.join(self.cache_dir, "compas-scores-two-years.csv"))
        return df


class GermanDataSource(DataSource):
    def __init__(self, resources=GERMAN_RESOURCES,
                 preprocess_fn=preprocess_german, **kwargs):
        super().__init__(resources=resources, preprocess_fn=preprocess_fn,
                         **kwargs)

    def _load_data(self) -> pd.DataFrame:
        df = pd.read_csv(os.path.join(self.cache_dir, "german.data"),
                         sep=" ", header=None)
        return df


class DiabetesReadmissionDataSource(DataSource):
    def __init__(self, resources=DIABETES_READMISSION_RESOURCES,
                 preprocess_fn=preprocess_diabetes_readmission, **kwargs):
        super().__init__(resources=resources, preprocess_fn=preprocess_fn,
                         **kwargs)

    def _load_data(self) -> pd.DataFrame:
        # unzip the file
        zip_fp = os.path.join(self.cache_dir, "dataset_diabetes.zip")
        with zipfile.ZipFile(zip_fp, 'r') as zf:
            zf.extractall(self.cache_dir)
        # read the dataframe
        df = pd.read_csv(os.path.join(self.cache_dir, "dataset_diabetes",
                                      "diabetic_data.csv"),
                         na_values="?",
                         low_memory=False)
        return df


class CommunitiesAndCrimeDataSource(DataSource):
    def __init__(self, resources=CANDC_RESOURCES,
                 preprocess_fn=preprocess_candc, **kwargs):
        super().__init__(resources=resources, preprocess_fn=preprocess_fn,
                         **kwargs)

    def _load_data(self) -> pd.DataFrame:
        df = pd.read_csv(os.path.join(self.cache_dir, 'communities.data'),
                         names=CANDC_INPUT_FEATURES)
        return df


class PhysioNetDataSource(DataSource):
    def __init__(self, preprocess_fn=preprocess_physionet, **kwargs):
        super().__init__(preprocess_fn=preprocess_fn, **kwargs)

    def _download_if_not_cached(self):
        # check if correct number of training files exist in cache dir
        root = os.path.join(self.cache_dir, "physionet.org", "files",
                            "challenge-2019", "1.0.0", "training")
        n_train_a = len(glob.glob(os.path.join(root, "training_setA", "*.psv")))
        n_train_b = len(glob.glob(os.path.join(root, "training_setB", "*.psv")))

        if (not n_train_a == 20336) or (not n_train_b == 20000):
            logging.info("downloading physionet training data. This could "
                         "take several minutes.")
            # download the training data
            cmd = "wget -r -N -c -np https://physionet.org/files/challenge" \
                  f"-2019/1.0.0/training/ -P={self.cache_dir}"
            utils.run_in_subprocess(cmd)
        else:
            logging.info(f"detected valid physionet training data at {root}; "
                         f"skipping download")
        return

    def _load_data(self) -> pd.DataFrame:
        root = os.path.join(self.cache_dir, "physionet.org", "files",
                            "challenge-2019", "1.0.0", "training")
        logging.info("reading physionet data files.")
        train_a_files = glob.glob(os.path.join(root, "training_setA", "*.psv"))
        df_a = pd.concat(pd.read_csv(x, delimiter="|") for x in train_a_files)
        train_b_files = glob.glob(os.path.join(root, "training_setB", "*.psv"))
        df_b = pd.concat(pd.read_csv(x, delimiter="|") for x in train_b_files)
        logging.info("done reading physionet data files.")
        df_a["set"] = "a"
        df_b["set"] = "b"
        df = pd.concat((df_a, df_b))
        df.reset_index(drop=True, inplace=True)
        return df


class MIMICExtractDataSource(OfflineDataSource):

    def __init__(self, task: str = "los_3",
                 preprocess_fn=preprocess_mimic_extract,
                 static_features=MIMIC_EXTRACT_STATIC_FEATURES.names, **kwargs):
        # Note: mean label values in overall dataset:
        # mort_hosp 0.106123
        # mort_icu 0.071709
        # los_3 0.430296
        # los_7 0.077055

        if task not in ('mort_hosp', 'mort_icu', 'los_3', 'los_7'):
            raise NotImplementedError(f"task {task} is not implemented.")
        self.task = task
        self.static_features = static_features
        _preprocess_fn = partial(preprocess_fn, task=task,
                                 static_features=self.static_features)
        super().__init__(**kwargs, preprocess_fn=_preprocess_fn)

    def _load_data(self, gap_time_hrs=6, window_size_hrs=24) -> pd.DataFrame:
        """Load the data and apply any shared MIMIC-extract preprocessing
        with default parameters."""

        filename = os.path.join(self.cache_dir, 'all_hourly_data.h5')
        assert os.path.exists(
            filename), \
            f"""file {filename} does not exist; see the TableShift 
            instructions for  accessing/placing the MIMIC-extract dataset at 
            the expected location. The data file can be accessed at 
            https://storage.googleapis.com/mimic_extract/all_hourly_data.h5 
            after  obtaining access as described at 
            https://github.com/MLforHealth/MIMIC_Extract"""
        data_full_lvl2 = pd.read_hdf(filename, 'vitals_labs')
        statics = pd.read_hdf(filename, 'patients')

        # Extract/compute labels, retaining only the labels for the current task.
        Ys = statics[statics.max_hours > window_size_hrs + gap_time_hrs][
            ['mort_hosp', 'mort_icu', 'los_icu']]
        Ys['los_3'] = Ys['los_icu'] > 3
        Ys['los_7'] = Ys['los_icu'] > 7
        Ys = Ys[[self.task]]
        Ys = Ys.astype(int)

        # MIMIC-default filtering: keep only those observations where labels are known; keep
        # only those observations within the window size.
        lvl2 = data_full_lvl2[
            (data_full_lvl2.index.get_level_values('icustay_id').isin(
                set(Ys.index.get_level_values('icustay_id')))) &
            (data_full_lvl2.index.get_level_values(
                'hours_in') < window_size_hrs)]

        # Join data with labels and static features.
        df_out = lvl2.join(Ys, how="inner")
        df_out = df_out.join(statics[MIMIC_EXTRACT_STATIC_FEATURES.names])
        assert len(df_out) == len(lvl2), "Sanity check of labels join."
        return df_out


class HELOCDataSource(OfflineDataSource):
    """FICO Home Equity Line of Credit data source.

    To obtain data access, visit
    https://community.fico.com/s/explainable-machine-learning-challenge
    """

    def __init__(self, preprocess_fn=preprocess_heloc, **kwargs):
        super().__init__(preprocess_fn=preprocess_fn, **kwargs)

    def _load_data(self) -> pd.DataFrame:
        filename = os.path.join(self.cache_dir, "heloc_dataset_v1.csv")
        assert os.path.exists(
            filename), \
            f"""file {filename} does not exist; see the TableShift 
            instructions for  accessing/placing the FICO HELOC dataset at the 
            expected location. The data file can be accessed by filling out the
            data access agreement at
            https://community.fico.com/s/explainable-machine-learning-challenge
            """
        return pd.read_csv(filename)


class MetaMIMICDataSource(OfflineDataSource):
    """MetaMIMIC data source.

    The dataset must be manually derived from MIMIC using the scripts
    provided in https://github.com/ModelOriented/metaMIMIC .
    """

    def __init__(self, preprocess_fn=lambda x: x, **kwargs):
        super().__init__(preprocess_fn=preprocess_fn, **kwargs)

    def _load_data(self) -> pd.DataFrame:
        filename = os.path.join(self.cache_dir, "metaMIMIC.csv")
        assert os.path.exists(filename), \
            f"file {filename} does not exist. Ensure you have constructed the " \
            f"metaMIMIC dataset as described at " \
            f"https://github.com/ModelOriented/metaMIMIC and placed the " \
            f"resulting file at {filename} ."
        return pd.read_csv(filename)


class GrinstajnHFDataSource(DataSource):
    """Fetch a dataset from the Grinstajn benchmark from hugging face hub."""

    def __init__(self, dataset_name: str,
                 preprocess_fn=preprocess_grinsztain_datataset, **kwargs):
        self.dataset_name = dataset_name
        _preprocess_fn = partial(preprocess_fn, name=dataset_name)
        super().__init__(preprocess_fn=_preprocess_fn, **kwargs)

    def _load_data(self, hf_split: str = datasets.Split.TRAIN) -> pd.DataFrame:
        """
        Loading function with special logic to support Grinstajn benchmark.
        """

        # The HF 'data_files' csv names. For datasets which are in both the
        # 'numeric only' and 'numeric + categorical' benchmarks, we keep the
        # version in the 'numeric + categorical' benchmark only, since it has
        # a superset of the original features.
        CLF_NUM_DSETS = ['bank-marketing', 'Bioresponse', 'california',
                         'credit', 'Higgs', 'house_16H', 'jannis',
                         'MagicTelescope', 'MiniBooNE', 'pol']
        CLF_CAT_DSETS = ['albert', 'covertype',
                         'default-of-credit-card-clients',
                         'electricity', 'eye_movements', 'road-safety']

        if self.dataset_name in CLF_NUM_DSETS:
            # numeric-only datasets
            data_file = f"clf_num/{self.dataset_name}.csv"
        elif self.dataset_name in CLF_CAT_DSETS:
            # numeric + categorical datasets
            data_file = f"clf_cat/{self.dataset_name}.csv"
        else:
            raise ValueError(f"dataset {self.dataset_name} not supported.")
        dataset = datasets.load_dataset("inria-soda/tabular-benchmark",
                                        data_files=data_file,
                                        cache_dir=self.cache_dir)
        return dataset[hf_split].to_pandas()

    def _download_if_not_cached(self):
        """No-op, since datasets.load_dataset() implements this."""
        return


class ClickDataSource(KaggleCompetitionDataSource):
    def __init__(self, preprocess_fn=preprocess_click, **kwargs):
        super().__init__(preprocess_fn=preprocess_fn, **kwargs)

    def _load_data(self) -> pd.DataFrame:
        """Implements data-loading logic from CatBoost benchmark for click."""
        dataset_dir = os.path.join(self.cache_dir, self.kaggle_dataset_name)
        files = [os.path.join(dataset_dir, 'track2', f) for f in
                 ('training.txt',)]

        if any([not os.path.exists(f) for f in files]):
            # Unzip the data file, which is zipped inside the main .zip file.
            zip_fp = os.path.join(dataset_dir, 'track2.zip')
            logging.info(f'unzipping {zip_fp}')
            # where to unzip the file to
            unzip_dest = os.path.join(self.cache_dir, self.kaggle_dataset_name)
            with zipfile.ZipFile(zip_fp, 'r') as zf:
                zf.extractall(unzip_dest)

        utils.download_file(
            "https://raw.githubusercontent.com/catboost/benchmarks/master/quality_benchmarks/prepare_click/stratified_test_idx.txt",
            dataset_dir)
        utils.download_file(
            "https://github.com/catboost/benchmarks/raw/master/quality_benchmarks/prepare_click/stratified_train_idx.txt",
            dataset_dir)
        utils.download_file(
            "https://github.com/catboost/benchmarks/raw/master/quality_benchmarks/prepare_click/subsampling_idx.txt",
            dataset_dir)

        # Data reading code from
        logging.debug('parsing ids from file subsampling_idx.txt')
        with open(os.path.join(dataset_dir, "subsampling_idx.txt")) as fin:
            ids = list(map(int, fin.read().split()))

        logging.debug('reading training data')
        unique_ids = set(ids)
        data_strings = {}
        with open(os.path.join(dataset_dir, 'track2', 'training.txt')) as fin:
            for i, string in enumerate(fin):
                if i in unique_ids:
                    data_strings[i] = string

        data_rows = []
        for i in ids:
            data_rows.append(data_strings[i])

        data = pd.read_table(StringIO("".join(data_rows)), header=None).apply(
            np.float64)
        colnames = ['click',
                    'impression',
                    'url_hash',
                    'ad_id',
                    'advertiser_id',
                    'depth',
                    'position',
                    'query_id',
                    'keyword_id',
                    'title_id',
                    'description_id',
                    'user_id']
        data.columns = colnames

        # train_idx = pd.read_csv(
        #     os.path.join(dataset_dir, "stratified_train_idx.txt"), header=None)
        test_idx = pd.read_csv(
            os.path.join(dataset_dir, "stratified_test_idx.txt"), header=None)

        data["Split"] = "train"
        data["Split"].iloc[test_idx] = "test"

        return data


class KddCup2009DataSource(DataSource):
    def __init__(self, task_name: str, **kwargs):
        self.task_name = task_name
        self.name = task_name
        _resources = [
            "https://kdd.org/cupfiles/KDDCupData/2009/orange_small_train.data.zip",
            f"http://www.kdd.org/cupfiles/KDDCupData/2009/orange_small_train_{task_name}.labels",
        ]
        super().__init__(resources=_resources,
                         preprocess_fn=preprocess_appetency, **kwargs)

    def _load_data(self) -> pd.DataFrame:
        data_fp = os.path.join(self.cache_dir, "orange_small_train.data")

        if not os.path.exists(data_fp):
            zip_fp = os.path.join(self.cache_dir, "orange_small_train.data.zip")

            with zipfile.ZipFile(zip_fp, 'r') as zf:
                zf.extractall(self.cache_dir)

        data = pd.read_csv(data_fp, sep="\t")
        labels = -pd.read_csv(
            os.path.join(self.cache_dir,
                         f"orange_small_train_{self.task_name}.labels"),
            header=None)[0]
        labels = labels.replace({-1: 0})
        labels.name = "label"
        data = pd.concat((data, labels), axis=1)
        return data


class KickDataSource(KaggleCompetitionDataSource):
    def __init__(self, preprocess_fn=preprocess_kick, **kwargs):
        super().__init__(preprocess_fn=preprocess_fn, **kwargs)

    def _load_data(self) -> pd.DataFrame:
        fp = os.path.join(self.cache_dir, self.kaggle_dataset_name,
                          "training.csv")
        return pd.read_csv(fp)


class AutoMLBenchmarkDataSource(OfflineDataSource):
    def __init__(self, automl_benchmark_dataset_name: str, **kwargs):
        self.dataset_name = automl_benchmark_dataset_name
        _preprocess_fn = partial(
            preprocess_automl,
            automl_benchmark_dataset_name=automl_benchmark_dataset_name)
        super().__init__(preprocess_fn=_preprocess_fn, **kwargs)

    def _load_data(self) -> pd.DataFrame:
        from auto_mm_bench.datasets import dataset_registry
        train_dataset = dataset_registry.create(self.dataset_name, 'train')
        train_df = train_dataset.data
        train_df['Split'] = 'train'

        test_dataset = dataset_registry.create(self.dataset_name, 'test')
        test_df = test_dataset.data
        test_df['Split'] = 'test'

        return pd.concat((train_df, test_df))


class IrisDataSource(DataSource):
    def __init__(self, **kwargs):
        _resources = [
            "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"]
        super().__init__(resources=_resources,
                         preprocess_fn=lambda x: x,
                         **kwargs)

    def _load_data(self) -> pd.DataFrame:
        fp = os.path.join(self.cache_dir, "iris.data")
        df = pd.read_csv(fp, names=['Sepal_Length', 'Sepal_Width',
                                    'Petal_Length', 'Petal_Width', 'Class'])
        return df


class DryBeanDataSource(DataSource):
    def __init__(self, **kwargs):
        _resources = [
            "https://archive.ics.uci.edu/ml/machine-learning-databases/00602/DryBeanDataset.zip"]
        super().__init__(resources=_resources,
                         preprocess_fn=lambda x: x,
                         **kwargs)

    def _load_data(self) -> pd.DataFrame:
        excel_fp = os.path.join(self.cache_dir, "DryBeanDataset",
                                "Dry_Bean_Dataset.xlsx")
        if not os.path.exists(excel_fp):
            zip_fp = os.path.join(self.cache_dir, "DryBeanDataset.zip")
            # where to unzip the file to
            with zipfile.ZipFile(zip_fp, 'r') as zf:
                zf.extractall(self.cache_dir)

        return pd.read_excel(excel_fp)


class HeartDiseaseDataSource(DataSource):
    def __init__(self, **kwargs):
        _resources = [
            "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"]
        super().__init__(resources=_resources,
                         preprocess_fn=lambda x: x,
                         **kwargs)

    def _load_data(self) -> pd.DataFrame:
        df = pd.read_csv(
            os.path.join(self.cache_dir, "processed.cleveland.data"),
            names=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                   'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal',
                   'num'],
            na_values='?')
        return df


class WineCultivarsDataSource(DataSource):
    def __init__(self, **kwargs):
        _resources = [
            "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"]
        super().__init__(resources=_resources,
                         preprocess_fn=lambda x: x,
                         **kwargs)

    def _load_data(self) -> pd.DataFrame:
        df = pd.read_csv(
            os.path.join(self.cache_dir, "wine.data"),
            names=WINE_CULTIVARS_FEATURES.names)
        return df


class WineQualityDataSource(DataSource):
    def __init__(self, **kwargs):
        _resources = [
            "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
            "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
        ]
        super().__init__(resources=_resources,
                         preprocess_fn=lambda x: x,
                         **kwargs)

    def _load_data(self) -> pd.DataFrame:
        df_red = pd.read_csv(
            os.path.join(self.cache_dir, "winequality-red.csv"), sep=";")
        df_white = pd.read_csv(
            os.path.join(self.cache_dir, "winequality-white.csv"), sep=";")
        df_red["red_or_white"] = "red"
        df_white["red_or_white"] = "white"
        return pd.concat((df_red, df_white))


class RiceDataSource(DataSource):
    """Rice data source. Uses a non-UCI archive since the main data files
    have been removed from UCI (even though it is still listed as a top
    downloaded dataset). """

    def __init__(self,
                 preprocess_fn=lambda x: x,
                 **kwargs):
        super().__init__(preprocess_fn=preprocess_fn,
                         **kwargs)

    def _download_if_not_cached(self):
        # The file is no longer available on UCI, so we load it from another
        # source. This source uses a PHP request, so we make and follow that
        # to download the actual file.

        ext = ".zip"
        zip_fp = os.path.join(self.cache_dir, "rice" + ext)

        if not os.path.exists(zip_fp):
            url = "https://www.muratkoklu.com/datasets/vtdhnd01.php"
            response = requests.get(url)
            content = response.content

            with open(zip_fp, 'wb') as f:
                f.write(content)
                f.close()

    def _load_data(self) -> pd.DataFrame:
        excel_fp = os.path.join(self.cache_dir,
                                "Rice_Dataset_Commeo_and_Osmancik",
                                "Rice_Cammeo_Osmancik.xlsx")
        if not os.path.exists(excel_fp):
            zip_fp = os.path.join(self.cache_dir, "rice.zip")
            # where to unzip the file to
            with zipfile.ZipFile(zip_fp, 'r') as zf:
                zf.extractall(self.cache_dir)

        return pd.read_excel(excel_fp)


class BreastCancerDataSource(DataSource):
    def __init__(self, preprocess_fn=lambda x: x, **kwargs):
        _resources = [
            "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data",
        ]
        super().__init__(preprocess_fn=preprocess_fn,
                         resources=_resources,
                         **kwargs)

    def _load_data(self) -> pd.DataFrame:
        df = pd.read_csv(
            os.path.join(self.cache_dir, "wdbc.data"),
            names=["id", "diagnosis", "radius_mean", "texture_mean",
                   "perimeter_mean", "area_mean", "smoothness_mean",
                   "compactness_mean", "concavity_mean", "concave_points_mean",
                   "symmetry_mean", "fractal_dimension_mean", "radius_std",
                   "texture_std", "perimeter_std", "area_std", "smoothness_std",
                   "compactness_std", "concavity_std", "concave_points_std",
                   "symmetry_std", "fractal_dimension_std", "radius_worst",
                   "texture_worst", "perimeter_worst", "area_worst",
                   "smoothness_worst", "compactness_worst", "concavity_worst",
                   "concave_points_worst", "symmetry_worst",
                   "fractal_dimension_worst"])
        return df


class CarDataSource(DataSource):
    def __init__(self, preprocess_fn=lambda x: x, **kwargs):
        _resources = [
            "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"]
        super().__init__(preprocess_fn=preprocess_fn, resources=_resources,
                         **kwargs)

    def _load_data(self) -> pd.DataFrame:
        df = pd.read_csv(os.path.join(self.cache_dir, "car.data"),
                         names=['buying', 'maint', 'doors', 'persons',
                                'lug_boot', 'safety', 'class'])
        return df


class RaisinDataSource(DataSource):
    def __init__(self, preprocess_fn=lambda x: x, **kwargs):
        _resources = [
            "https://archive.ics.uci.edu/ml/machine-learning-databases/00617/Raisin_Dataset.zip"]
        super().__init__(preprocess_fn=preprocess_fn,
                         resources=_resources,
                         **kwargs)

    def _load_data(self) -> pd.DataFrame:
        excel_fp = os.path.join(self.cache_dir,
                                "Raisin_Dataset",
                                "Raisin_Dataset.xlsx")
        if not os.path.exists(excel_fp):
            zip_fp = os.path.join(self.cache_dir, "Raisin_Dataset.zip")
            # where to unzip the file to
            with zipfile.ZipFile(zip_fp, 'r') as zf:
                zf.extractall(self.cache_dir)
        return pd.read_excel(excel_fp)


class AbaloneDataSource(DataSource):
    def __init__(self, preprocess_fn=preprocess_abalone, **kwargs):
        _resources = [
            "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data"]
        super().__init__(preprocess_fn=preprocess_fn,
                         resources=_resources,
                         **kwargs)

    def _load_data(self) -> pd.DataFrame:
        df = pd.read_csv(os.path.join(self.cache_dir, "abalone.data"),
                         names=ABALONE_FEATURES.names)
        return df


class OttoProductsDataSource(KaggleCompetitionDataSource):
    def __init__(self, **kwargs):
        super().__init__(
            **kwargs,
            preprocess_fn=preprocess_otto,
            kaggle_dataset_name="otto-group-product-classification-challenge")

    def _load_data(self) -> pd.DataFrame:
        df = pd.read_csv(os.path.join(
            self.cache_dir,
            'otto-group-product-classification-challenge',
            'train.csv'))
        return df


class SfCrimeDataSource(KaggleCompetitionDataSource):
    def __init__(self, **kwargs):
        super().__init__(kaggle_dataset_name='sf-crime',
                         preprocess_fn=lambda x: x, **kwargs)

    @property
    def zip_file_name(self):
        return "train.csv.zip"

    def _load_data(self) -> pd.DataFrame:
        csv_fp = os.path.join(self.cache_dir, "sf-crime", "train.csv")
        df = pd.read_csv(csv_fp)
        return df


class PlasticcDataSource(KaggleCompetitionDataSource):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, kaggle_dataset_name='PLAsTiCC-2018',
                         preprocess_fn=lambda x: x)

    @property
    def zip_file_name(self):
        return "training_set.csv.zip"

    def _download_kaggle_data(self):
        """Download the data from Kaggle.

        All data files are 40GB, but the training set is only 60MB, so we download
        only the training set.
        """
        self._check_creds()

        # Download using Kaggle CLI.
        try:
            cmds = ("kaggle competitions download " \
                    f"{self.kaggle_dataset_name} -f training_set.csv " \
                    f"-p {self.cache_dir}",
                    "kaggle competitions download " \
                    f"{self.kaggle_dataset_name} -f training_set_metadata.csv " \
                    f"-p {self.cache_dir}"
                    )
            for cmd in cmds:
                utils.run_in_subprocess(cmd)
        except Exception as e:
            logging.warning("exception when downloading data; maybe you"
                            "need to visit the competition page on kaggle"
                            "and agree to the terms of the competition?")
            raise (e)
        return

    def _load_data(self) -> pd.DataFrame:
        df = pd.read_csv(os.path.join(self.cache_dir,
                                      self.kaggle_dataset_name,
                                      "training_set.csv"))
        meta_df = pd.read_csv(os.path.join(self.cache_dir,
                                           "training_set_metadata.csv"))
        df = df.merge(meta_df, on="object_id", how="inner")
        return df


class WalmartDataSource(KaggleCompetitionDataSource):
    def __init__(self, **kwargs):
        super().__init__(
            kaggle_dataset_name="walmart-recruiting-trip-type-classification",
            preprocess_fn=preprocess_walmart, **kwargs)

    def _load_data(self) -> pd.DataFrame:
        csv_fp = os.path.join(self.cache_dir, self.kaggle_dataset_name,
                              "train.csv")
        if not os.path.exists(csv_fp):
            zip_fp = os.path.join(self.cache_dir, self.kaggle_dataset_name,
                                  "train.csv.zip")
            with zipfile.ZipFile(zip_fp, 'r') as zf:
                zf.extractall(os.path.join(self.cache_dir,
                                           self.kaggle_dataset_name))
        df = pd.read_csv(csv_fp)
        return df


class TradeShiftDataSource(KaggleCompetitionDataSource):
    def __init__(self, label_colname: str = 'y33', **kwargs):
        self.label_colname = label_colname
        super().__init__(
            kaggle_dataset_name="tradeshift-text-classification",
            preprocess_fn=lambda x: x, **kwargs)

    def _load_data(self) -> pd.DataFrame:
        train_data_fp = os.path.join(self.cache_dir,
                                     self.kaggle_dataset_name,
                                     "train.csv.gz")
        train_labels_fp = os.path.join(self.cache_dir,
                                       self.kaggle_dataset_name,
                                       "trainLabels.csv.gz")
        logging.debug(f'reading training data from {train_data_fp}')
        with gzip.open(train_data_fp) as f:
            train_data_df = pd.read_csv(f)

        logging.debug(f'reading train labels from {train_labels_fp}')
        with gzip.open(train_labels_fp) as f:
            train_labels = pd.read_csv(f)

        train_data_df[self.label_colname] = train_labels[self.label_colname]

        return train_data_df


class SchizophreniaDataSource(KaggleCompetitionDataSource):
    def __init__(self, **kwargs):
        super().__init__(preprocess_fn=lambda x: x,
                         kaggle_dataset_name="mlsp-2014-mri",
                         **kwargs)

    def _load_data(self) -> pd.DataFrame:
        train_data_dir = os.path.join(self.cache_dir,
                                      self.kaggle_dataset_name)
        train_fnc_fp = os.path.join(train_data_dir, "train_FNC.csv")
        train_labels_fp = os.path.join(train_data_dir, "train_labels.csv")
        train_sbm_fp = os.path.join(train_data_dir, "train_SBM.csv")
        if any(not os.path.exists(f) for f in
               (train_fnc_fp, train_labels_fp, train_sbm_fp)):
            zip_fp = os.path.join(self.cache_dir, self.kaggle_dataset_name,
                                  "Train.zip")
            with zipfile.ZipFile(zip_fp, 'r') as zf:
                zf.extractall(os.path.join(self.cache_dir,
                                           self.kaggle_dataset_name))

        train_fnc = pd.read_csv(train_fnc_fp)
        train_labels = pd.read_csv(train_labels_fp)
        train_sbm = pd.read_csv(train_sbm_fp)
        df = train_fnc.merge(train_labels, on="Id").merge(train_sbm, on="Id")
        return df


class TitanicDataSource(KaggleCompetitionDataSource):
    def __init__(self, **kwargs):
        super().__init__(kaggle_dataset_name='titanic',
                         preprocess_fn=lambda x: x,
                         **kwargs)

    def _load_data(self) -> pd.DataFrame:
        df = pd.read_csv(os.path.join(self.cache_dir,
                                      self.kaggle_dataset_name,
                                      "train.csv"))
        return df


class SantanderTransactionDataSource(KaggleCompetitionDataSource):
    def __init__(self, **kwargs):
        super().__init__(
            kaggle_dataset_name='santander-customer-transaction-prediction',
            preprocess_fn=lambda x: x,
            **kwargs)

    def _load_data(self) -> pd.DataFrame:
        df = pd.read_csv(os.path.join(self.cache_dir,
                                      self.kaggle_dataset_name,
                                      "train.csv"))
        return df


class HomeCreditDefaultDataSource(KaggleCompetitionDataSource):

    def __init__(self, **kwargs):
        super().__init__(kaggle_dataset_name='home-credit-default-risk',
                         preprocess_fn=lambda x: x,
                         **kwargs)

    @property
    def zip_file_name(self):
        return "application_train.csv.zip"

    def _download_kaggle_data(self):
        """Download the data from Kaggle.

        All data files are 40GB, but the training set is only 60MB, so we download
        only the training set.
        """
        self._check_creds()

        # Download using Kaggle CLI.
        try:
            cmd = "kaggle competitions download " \
                  f"{self.kaggle_dataset_name} -f application_train.csv " \
                  f"-p {self.cache_dir}"

            utils.run_in_subprocess(cmd)
        except Exception as e:
            logging.warning("exception when downloading data; maybe you"
                            "need to visit the competition page on kaggle"
                            "and agree to the terms of the competition?")
            raise e
        return

    def _load_data(self) -> pd.DataFrame:
        df = pd.read_csv(os.path.join(self.cache_dir,
                                      self.kaggle_dataset_name,
                                      "application_train.csv"))
        return df


class IeeFraudDetectionDataSource(KaggleCompetitionDataSource):
    def __init__(self, **kwargs):
        super().__init__(kaggle_dataset_name='ieee-fraud-detection',
                         preprocess_fn=apply_column_missingness_threshold,
                         **kwargs)

    def _load_data(self) -> pd.DataFrame:
        logging.debug(f'reading data for {self.kaggle_dataset_name}')
        df = pd.read_csv(os.path.join(self.cache_dir,
                                      self.kaggle_dataset_name,
                                      "train_transaction.csv"))
        identity = pd.read_csv(os.path.join(self.cache_dir,
                                            self.kaggle_dataset_name,
                                            "train_identity.csv"))
        logging.debug(f"merging data for {self.kaggle_dataset_name}")
        df = df.merge(identity, on="TransactionID", how="left",
                      suffixes=(None, "_y"))
        df.drop(columns=[c for c in df.columns if c.endswith("_y")],
                inplace=True)
        return df


class SafeDriverPredictionDataSource(KaggleCompetitionDataSource):
    def __init__(self, **kwargs):
        super().__init__(
            kaggle_dataset_name='porto-seguro-safe-driver-prediction',
            preprocess_fn=lambda x: x,
            **kwargs)

    def _load_data(self) -> pd.DataFrame:
        df = pd.read_csv(os.path.join(self.cache_dir,
                                      self.kaggle_dataset_name,
                                      "train.csv"))
        return df


class SantanderCustomerSatisfactionDataSource(KaggleCompetitionDataSource):
    def __init__(self, **kwargs):
        super().__init__(
            kaggle_dataset_name='santander-customer-satisfaction',
            preprocess_fn=lambda x: x,
            **kwargs)

    def _load_data(self) -> pd.DataFrame:
        df = pd.read_csv(os.path.join(self.cache_dir,
                                      self.kaggle_dataset_name,
                                      "train.csv"))
        return df


class AmexDefaultDataSource(KaggleCompetitionDataSource):
    def __init__(self, **kwargs):
        super().__init__(
            kaggle_dataset_name='amex-default-prediction',
            preprocess_fn=apply_column_missingness_threshold,
            **kwargs)

    @property
    def zip_file_name(self):
        return "train_data.csv.zip"

    def _download_kaggle_data(self):
        self._check_creds()

        # Download using Kaggle CLI.
        cmds = ("kaggle competitions download " \
                f"{self.kaggle_dataset_name} -f train_data.csv " \
                f"-p {self.cache_dir}",
                "kaggle competitions download " \
                f"{self.kaggle_dataset_name} -f train_labels.csv " \
                f"-p {self.cache_dir}"
                )
        for cmd in cmds:
            res = utils.run_in_subprocess(cmd)
            if res.returncode != 0:
                raise KaggleDownloadError

        return

    def _load_data(self) -> pd.DataFrame:
        df = pd.read_csv(os.path.join(self.cache_dir,
                                      self.kaggle_dataset_name,
                                      "train_data.csv"))
        labels_fp = os.path.join(self.cache_dir, self.kaggle_dataset_name,
                                 "train_labels.csv")
        if not os.path.exists(labels_fp):
            zip_fp = os.path.join(self.cache_dir, "train_labels.csv.zip")
            with zipfile.ZipFile(zip_fp, 'r') as zf:
                zf.extractall(os.path.join(self.cache_dir,
                                           self.kaggle_dataset_name))

        labels = pd.read_csv(labels_fp)
        df = df.merge(labels, on='customer_ID', how='inner')
        return df


class AdFraudDataSource(KaggleCompetitionDataSource):
    def __init__(self, **kwargs):
        super().__init__(
            kaggle_dataset_name='talkingdata-adtracking-fraud-detection',
            preprocess_fn=lambda x: x,
            **kwargs)

    def _load_data(self) -> pd.DataFrame:
        df = pd.read_csv(os.path.join(self.cache_dir,
                                      self.kaggle_dataset_name,
                                      "train.csv"))

        # Drop this column since it perfectly predicts the label.
        df.drop(columns=['attributed_time'], inplace=True)
        return df


class AssistmentsDataSource(KaggleDataSource):
    def __init__(self, **kwargs):
        super().__init__(
            kaggle_dataset_name="nicolaswattiez/skillbuilder-data-2009-2010",
            preprocess_fn=tableshift.datasets.preprocess_assistments, **kwargs)

    def _load_data(self) -> pd.DataFrame:
        logging.info(
            "reading assistments data (can be slow due to large file size)")
        # TODO(jpgard): uncomment below to use full-width dataset after testing.
        # df = pd.read_csv(os.path.join(
        #     self.cache_dir,
        #     self.kaggle_dataset_name,
        #     "2012-2013-data-with-predictions-4-final.csv"))
        # # # write out a tiny version of assistments datasets
        # import ipdb;
        # ipdb.set_trace()
        # df[tableshift.datasets.ASSISTMENTS_FEATURES.names].to_feather(
        #     os.path.join(self.cache_dir, "assistments-subset.feather"))
        df = pd.read_feather(os.path.join(self.cache_dir,
                                          "assistments-subset.feather"))
        logging.info("finished reading data")
        return df


class CollegeScorecardDataSource(KaggleDataSource):
    def __init__(self, **kwargs):
        super().__init__(
            kaggle_dataset_name="kaggle/college-scorecard",
            preprocess_fn=tableshift.datasets.preprocess_college_scorecard,
            **kwargs)

    def _load_data(self) -> pd.DataFrame:
        return pd.read_csv(os.path.join(self.cache_dir,
                                        self.kaggle_dataset_name,
                                        "Scorecard.csv"))

import logging
from typing import Optional, Dict, Any, Union

from tableshift.configs.experiment_defaults import DEFAULT_RANDOM_STATE
from tableshift.configs.benchmark_configs import BENCHMARK_CONFIGS
from tableshift.configs.non_benchmark_configs import NON_BENCHMARK_CONFIGS
from .tabular_dataset import TabularDataset, DatasetConfig, CachedDataset
from .features import PreprocessorConfig
from .splitter import RandomSplitter

EXPERIMENT_CONFIGS = {
    **BENCHMARK_CONFIGS,
    **NON_BENCHMARK_CONFIGS
}


def get_dataset(name: str, cache_dir: str = "tmp",
                preprocessor_config: Optional[
                    PreprocessorConfig] = None,
                initialize_data: bool = True,
                use_cached: bool = False,
                **kwargs) -> Union[TabularDataset, CachedDataset]:
    """Helper function to fetch a dataset.

    Args:
        name: the dataset name.
        cache_dir: the cache directory to use. TableShift will check for cached
            data files here before downloading.
        preprocessor_config: optional Preprocessor to override the default
            preprocessor config. If using the TableShift benchmark, it is
            recommended to leave this as None to use the default preprocessor.
        initialize_data: passed to TabularDataset constructor.
        use_cached: whether to used cached dataset.
        kwargs: optional kwargs to be passed to TabularDataset; these will
            override their respective kwargs in the experiment config.
        """
    assert name in EXPERIMENT_CONFIGS.keys(), \
        f"Dataset name {name} is not available; choices are: " \
        f"{sorted(EXPERIMENT_CONFIGS.keys())}"

    expt_config = EXPERIMENT_CONFIGS[name]
    dataset_config = DatasetConfig(cache_dir=cache_dir)
    tabular_dataset_kwargs = expt_config.tabular_dataset_kwargs
    if "name" not in tabular_dataset_kwargs:
        tabular_dataset_kwargs["name"] = name

    if preprocessor_config is None:
        preprocessor_config = expt_config.preprocessor_config

    if not use_cached:
        dset = TabularDataset(
            config=dataset_config,
            splitter=expt_config.splitter,
            grouper=kwargs.get("grouper", expt_config.grouper),
            preprocessor_config=preprocessor_config,
            initialize_data=initialize_data,
            **tabular_dataset_kwargs)
    else:
        dset = CachedDataset(config=dataset_config,
                             splitter=expt_config.splitter,
                             grouper=kwargs.get("grouper", expt_config.grouper),
                             preprocessor_config=preprocessor_config,
                             initialize_data=initialize_data,
                             name=name)
    return dset


def get_iid_dataset(name: str, cache_dir: str = "tmp",
                    val_size: float = 0.1,
                    test_size: float = 0.25,
                    random_state: int = DEFAULT_RANDOM_STATE,
                    preprocessor_config: Optional[
                        PreprocessorConfig] = None,
                    initialize_data: bool = True,
                    use_cached: bool = False,
                    tabular_dataset_kwargs: Optional[Dict[str, Any]] = None,
                    **kwargs,

                    ) -> Union[TabularDataset, CachedDataset]:
    """Helper function to fetch an IID dataset.

    This fetches a version of the TableShift benchmark dataset but *witihout*
    a domain split. This is mostly for testing or exploring non-domain-robust
    learning methods.

    Args:
        name: the dataset name.
        cache_dir: the cache directory to use. TableShift will check for cached
            data files here before downloading.
        val_size: fraction of dataset to use for validation split.
        test_size: fraction of dataset to use for test split.
        random_state: integer random state to use for splitting,
            for reproducibility.
        preprocessor_config: optional Preprocessor to override the default
            preprocessor config. If using the TableShift benchmark, it is
            recommended to leave this as None to use the default preprocessor.
        initialize_data: passed to TabularDataset constructor.
        use_cached: if True, load a cached dataset from cache_dir
            with specified uid.
        uid: uid to use for the cached dataset. Not used when use_cached=False.
        tabular_dataset_kwargs: optional overrides for tabular dataset kwargs.
        kwargs: optional kwargs to be passed to TabularDataset; these will
            override their respective kwargs in the experiment config.
        """
    assert name in EXPERIMENT_CONFIGS.keys(), \
        f"Dataset name {name} is not available; choices are: " \
        f"{sorted(EXPERIMENT_CONFIGS.keys())}"

    expt_config = EXPERIMENT_CONFIGS[name]
    dataset_config = DatasetConfig(cache_dir=cache_dir)

    _tabular_dataset_kwargs = expt_config.tabular_dataset_kwargs
    if tabular_dataset_kwargs:
        _tabular_dataset_kwargs.update(tabular_dataset_kwargs)
    if "name" not in _tabular_dataset_kwargs:
        _tabular_dataset_kwargs["name"] = name

    if preprocessor_config is None:
        preprocessor_config = expt_config.preprocessor_config

    if not use_cached:

        dset = TabularDataset(
            config=dataset_config,
            splitter=RandomSplitter(val_size=val_size,
                                    random_state=random_state,
                                    test_size=test_size),
            grouper=kwargs.get("grouper", expt_config.grouper),
            preprocessor_config=preprocessor_config,
            initialize_data=initialize_data,
            **_tabular_dataset_kwargs)
    else:

        logging.info(f"loading cached data from {cache_dir}")
        dset = CachedDataset(config=dataset_config,
                             name=name,
                             initialize_data=initialize_data,
                             preprocessor_config=preprocessor_config)

    return dset

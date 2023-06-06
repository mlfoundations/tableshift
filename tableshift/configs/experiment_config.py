from dataclasses import dataclass
from typing import Union, Dict, Any

from tableshift.core import Grouper, PreprocessorConfig, Splitter


@dataclass
class ExperimentConfig:
    splitter: Splitter
    grouper: Union[Grouper, None]
    preprocessor_config: PreprocessorConfig
    tabular_dataset_kwargs: Dict[str, Any]

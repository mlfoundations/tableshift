import fairlearn.reductions
import pandas as pd
import ray.data.dataset
import sklearn
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import warnings
from typing import Any, Dict, Optional, List

from ray.air import session
from ray.air.checkpoint import Checkpoint
from ray.air.config import RunConfig, ScalingConfig
from ray.train.trainer import BaseTrainer, GenDataset


class ExponentiatedGradient(fairlearn.reductions.ExponentiatedGradient):
    """Custom class to allow for scikit-learn-compatible interface.

    Specifically, this method takes (and ignores) a sample_weights
    parameter to its .fit() method; otherwise identical to
    fairlearn.ExponentiatedGradient.
    """

    def __init__(self, constraints, eps: float = 0.01, eta0: float = 2.,
                 base_estimator_cls=xgb.XGBClassifier,
                 **kwargs):
        estimator = base_estimator_cls(**kwargs)
        super().__init__(estimator=estimator,
                         constraints=constraints,
                         eps=eps,
                         eta0=eta0)

        # The LabelEncoder is used to ensure sensitive features are of
        # numerical type (not string/categorical).
        self.le = LabelEncoder()

    def fit(self, X: pd.DataFrame, y: pd.Series,
            sample_weight=None, **kwargs):
        del sample_weight

        # Fetch and check the domain labels
        assert "d" in kwargs, "require 'd', array with domain labels."
        d = kwargs.pop("d")
        assert isinstance(d, pd.Series)

        # Numerically encode the sensitive attribute; fairlearn.reductions
        # does not accept categorical/string-type data.
        domains_enc = self.le.fit_transform(d.values)

        with warnings.catch_warnings():
            # Filter FutureWarnings raised by fairlearn that overwhelm the
            # console output.
            warnings.filterwarnings("ignore", category=FutureWarning)

            super().fit(X.values, y.values, sensitive_features=domains_enc,
                        **kwargs)

    def predict(self, X, random_state=None):
        return super().predict(X)

    def predict_proba(self, X):
        """Alias to _pmf_predict(). Note that this tends to return very
        close to 'hard' predictions (probabilities close to 0/1), which don't
        perform well for metrics like cross-entropy."""
        return super()._pmf_predict(X)


class ExponentiatedGradientTrainer(BaseTrainer):
    def __init__(self, *,
                 datasets: Dict[str, GenDataset],
                 label_column: str,
                 domain_column: str,
                 feature_columns: List[str],
                 params: Dict[str, Any],
                 scaling_config: Optional[ScalingConfig] = None,
                 run_config: Optional[RunConfig] = None,
                 resume_from_checkpoint: Optional[Checkpoint] = None,
                 **train_kwargs):
        raise NotImplementedError("ExponentiatedGradientTrainer is not "
                                  "implemented.")
        self.params = params
        self.label_column = label_column
        self.domain_column = domain_column
        self.feature_columns = feature_columns
        self.model = None
        super().__init__(
            scaling_config=scaling_config,
            run_config=run_config,
            datasets=datasets,
            preprocessor=None,
            resume_from_checkpoint=resume_from_checkpoint,
        )

    def setup(self) -> None:
        self.model = ExponentiatedGradient(**self.params)

    def training_loop(self) -> None:
        train_dataset = self.datasets["train"]

        def prepare_dataset(ds):
            y = ds.map(lambda x: x[self.label_column])
            X = ds.map(lambda x: x[self.feature_columns])
            d = ds.map(lambda x: x[self.domain_column])
            return X, y, d

        X, y, d = prepare_dataset(train_dataset)
        # TODO(jpgard): will need to bring full dataset in-memory,
        # convert to pandas, and train.
        self.model.fit(X=X, y=y, d=d)
        X_val, y_val, d_val = prepare_dataset(self.datasets["validation"])
        y_hat_val = self.model.predict(X_val)
        acc_val = sklearn.metrics.accuracy_score(y_val, y_hat_val)
        session.report(dict(validation_accuracy=acc_val))
        return

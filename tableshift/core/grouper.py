from dataclasses import dataclass
import logging
from typing import Mapping, Sequence, Any, List

import pandas as pd
from sklearn.compose import ColumnTransformer


@dataclass
class Grouper:
    features_and_values: Mapping[str, Sequence[Any]]
    drop: bool
    transformer: ColumnTransformer = None

    @property
    def features(self) -> List[str]:
        return list(self.features_and_values.keys())

    def _check_inputs(self, data: pd.DataFrame):
        """Check inputs to the transform function."""
        for c in self.features:
            assert c in data.columns, \
                f"data does not contain grouping feature {c}"
            data_vals = set(data[c].unique().tolist())
            group_vals = set(self.features_and_values[c])
            intersection = data_vals.intersection(group_vals)
            assert len(intersection), \
                f"None of the specified grouping values {group_vals} " \
                f"are in column {c} values {data_vals}. Do the grouping " \
                f"values have the same type as the column type {data[c].dtype}?"

            missing_ood_vals = set(group_vals) - set(data_vals)
            if len(missing_ood_vals):
                logging.warning(f"values {list(missing_ood_vals)} specified "
                      f"in Grouper split but not  present in the data.")
            return

    def _check_transformed(self, data: pd.DataFrame):
        """Check the outputs of the transform function."""
        for c in self.features:
            vals = data[c].unique()
            if len(vals) < 2:
                raise ValueError(f"[ERROR] column {c} contains only one "
                                 f"unique value after transformation {vals}")

        # Print a summary of the counts after grouping.
        logging.debug("overall counts after grouping:")
        if len(self.features) == 1:
            logging.info(data[self.features[0]].value_counts())
        else:
            row_feat = self.features[0]
            col_feats = self.features[1:]
            xt = pd.crosstab(data[row_feat].squeeze(),
                             data[col_feats].squeeze(),
                             dropna=False)
            logging.debug(xt)

    def _group_column(self, x: pd.Series, vals: Sequence) -> pd.Series:
        """Apply a grouping to a column, retuning a binary numeric Series."""
        # Ensure types are the same by casting vals to the same type as x.
        tmp = pd.Series(vals).astype(x.dtype)
        return x.isin(tmp).astype(int)

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        self._check_inputs(data)
        for c in self.features:
            data[c] = self._group_column(data[c], self.features_and_values[c])
        self._check_transformed(data)
        return data

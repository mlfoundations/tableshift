"""
Implements the 'Weighted ERM for Covariate Shift' model described in
'Probabilistic Machine Learning: Advanced Topics' by Kevin Murphy
(draft accessed 11/16/22 via https://github.com/probml/pml2-book).
"""
import logging
import numpy as np
import pandas as pd
from sklearn import linear_model


class WeightedCovariateShiftClassifier:
    def __init__(self, C_domain: float, C_discrim: float):
        # Used to predict weights for training examples
        self.domain_classifier = linear_model.LogisticRegression(C=C_domain)
        # Used to predict labels
        self.discriminator = linear_model.LogisticRegression(C=C_discrim)

    def predict_importance_weights(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values

        weights = X @ np.squeeze(self.domain_classifier.coef_, 0)
        return weights

    def fit(self, X_id, y_id, X_ood):
        assert X_id.shape[1] == X_ood.shape[1], "Incompatible input shapes."
        # Fit the domain classifier
        if isinstance(X_id, pd.DataFrame):
            X = pd.concat((X_id, X_ood), axis=0)
        else:
            X = np.row_stack((X_id, X_ood))
        y = np.concatenate((np.ones((len(X_id),)), -np.ones(len(X_ood), )))
        logging.info("fitting domain classifier")
        self.domain_classifier.fit(X, y)

        # Fit the discriminator
        logging.info("fitting discriminator")
        id_sample_weights = self.predict_importance_weights(X_id)
        self.discriminator.fit(X_id, y_id, sample_weight=id_sample_weights)

    def predict(self, X) -> np.ndarray:
        return self.discriminator.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        return self.discriminator.predict_proba(X)

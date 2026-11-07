from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler


NON_FEATURE_COLS = {
    "Unnamed: 0.1", "Unnamed: 0", "sentence", "outlet", "topic", "type", "num_sent",
    "biased_words4", "text", "text_low", "pos", "lemma", "lemma_low", "tag", "dep",
    "label3", "label4", "label5", "ne_label"
}


def extract_numeric_features(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """Extract numeric features matrix from DataFrame, dropping text and meta columns."""
    cols = [c for c in df.columns if c not in NON_FEATURE_COLS and pd.api.types.is_numeric_dtype(df[c])]
    X = df[cols].fillna(0).values
    return X, cols


class MLBaseModel(BaseEstimator, ClassifierMixin):
    """Wrapper base class for ML baseline classifiers supporting pandas input."""

    def __init__(self):
        self.feature_names_: List[str] = []
        self.scaler = None
        self.model = None

    def _prepare_X(self, X: Union[pd.DataFrame, np.ndarray], fit: bool = False) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            if fit:
                X_mat, self.feature_names_ = extract_numeric_features(X)
            else:
                cols = self.feature_names_ if self.feature_names_ else [
                    c for c in X.columns if c not in NON_FEATURE_COLS and pd.api.types.is_numeric_dtype(X[c])
                ]
                X_mat = X[cols].fillna(0).values
        else:
            X_mat = np.nan_to_num(X, nan=0.0)

        if self.scaler is not None:
            if fit:
                X_mat = self.scaler.fit_transform(X_mat)
            else:
                X_mat = self.scaler.transform(X_mat)
        return X_mat

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]):
        X_mat = self._prepare_X(X, fit=True)
        y_vec = y.values if isinstance(y, pd.Series) else np.asarray(y)
        self.model.fit(X_mat, y_vec)
        self.classes_ = self.model.classes_
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        X_mat = self._prepare_X(X, fit=False)
        return self.model.predict(X_mat)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        X_mat = self._prepare_X(X, fit=False)
        return self.model.predict_proba(X_mat)


class WeightedLogisticRegressionBaseline(MLBaseModel):
    """
    Weighted Logistic Regression Baseline (Table 6 in paper).
    Applies inverse frequency / class weighting to account for the ~9.5% class imbalance.
    """

    def __init__(self, class_weight: Dict[int, int] = None, max_iter: int = 1000, random_state: int = 42):
        super().__init__()
        self.class_weight = class_weight or {0: 10, 1: 90}
        self.max_iter = max_iter
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = LogisticRegression(
            class_weight=self.class_weight,
            max_iter=self.max_iter,
            random_state=self.random_state,
            solver="lbfgs",
        )


class ComplementNaiveBayesBaseline(MLBaseModel):
    """
    Complement Naive Bayes Baseline (Table 6 in paper).
    Specifically designed to handle severe class imbalance in text and NLP classification.
    """

    def __init__(self):
        super().__init__()
        self.scaler = MinMaxScaler()
        self.model = ComplementNB()


class RandomForestBaseline(MLBaseModel):
    """
    Random Forest Classifier Baseline (Table 6 in paper).
    Uses an ensemble of decision trees with cost-sensitive class weighting.
    """

    def __init__(self, n_estimators: int = 100, class_weight: Dict[int, int] = None, random_state: int = 42):
        super().__init__()
        self.n_estimators = n_estimators
        self.class_weight = class_weight or {0: 10, 1: 90}
        self.random_state = random_state
        self.scaler = None
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            class_weight=self.class_weight,
            random_state=self.random_state,
            n_jobs=-1,
        )


def get_ml_baselines(random_state: int = 42) -> Dict[str, MLBaseModel]:
    """Factory creating supervised ML baselines corresponding to Table 6."""
    return {
        "Weighted Logistic Regression": WeightedLogisticRegressionBaseline(random_state=random_state),
        "Complement Naive Bayes": ComplementNaiveBayesBaseline(),
        "Random Forest (Weighted)": RandomForestBaseline(random_state=random_state),
    }

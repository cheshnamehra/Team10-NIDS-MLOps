"""Pickle-stable preprocessing transformers shared by training and serving."""
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd

class DataFrameMedianImputer(BaseEstimator, TransformerMixin):
    """Median imputation that preserves DataFrame column names."""
    def fit(self, X, y=None):
        self.columns_ = list(X.columns)
        self.imputer_ = SimpleImputer(strategy="median", keep_empty_features=True)
        self.imputer_.fit(X)
        return self
    def transform(self, X):
        frame = pd.DataFrame(X, columns=self.columns_)
        values = self.imputer_.transform(frame)
        return pd.DataFrame(values, columns=self.columns_, index=frame.index)

class NIDSFeatureEngineer(BaseEstimator, TransformerMixin):
    """Training-fitted log transforms and ordinal Flow Duration quartile bin."""
    def __init__(self, skew_threshold=2.0):
        self.skew_threshold = skew_threshold

    def fit(self, X, y=None):
        X = self._frame(X)
        self.columns_ = list(X.columns)
        self.skewed_columns_ = [
            c for c in X.columns
            if X[c].min() >= 0 and abs(X[c].skew()) > self.skew_threshold
        ]
        self.flow_col_ = next((c for c in X.columns if c.strip().lower() == "flow duration"), None)
        self.bin_edges_ = None
        if self.flow_col_:
            _, edges = pd.qcut(X[self.flow_col_], q=4, retbins=True, duplicates="drop")
            edges = np.unique(edges.astype(float))
            if len(edges) >= 2:
                edges[0], edges[-1] = -np.inf, np.inf
                self.bin_edges_ = edges
        return self

    def transform(self, X):
        X = self._frame(X).copy()
        X = X.reindex(columns=self.columns_)
        for col in self.skewed_columns_:
            X[col] = np.log1p(X[col].clip(lower=0))
        if self.flow_col_ and self.bin_edges_ is not None:
            X["Flow_Duration_Bin"] = pd.cut(
                X[self.flow_col_], bins=self.bin_edges_, labels=False,
                include_lowest=True
            ).fillna(0).astype(float)
        return X.to_numpy(dtype=float)

    @staticmethod
    def _frame(X):
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(X)


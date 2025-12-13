import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


# ==========================
# Custom Frequency Encoder
# ==========================
class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """
    Encodes a single categorical column using frequency/count encoding.
    Input shape  : (n_samples, 1)
    Output shape : (n_samples, 1)
    """
    def fit(self, X, y=None):
        X = pd.Series(X.ravel())
        self.freq_ = X.value_counts().to_dict()
        return self

    def transform(self, X):
        X = pd.Series(X.ravel())
        encoded = X.map(self.freq_).fillna(0)
        return encoded.to_numpy().reshape(-1, 1)


# ==========================
# Config
# ==========================
@dataclass
class DataTransformationConfig:
    artifacts_dir: Path = Path("artifacts")
    preprocessor_file_path: Path = artifacts_dir / "preprocessor.pkl"


# ==========================
# Data Transformation
# ==========================
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(
        self,
        num_col: List[str],
        cat_col: List[str],
        district_col: str
    ) -> ColumnTransformer:
        """
        Creates and returns the ColumnTransformer object.
        """
        try:
            logging.info("Creating preprocessing pipelines")

            # ---------- Numeric Pipeline ----------
            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            # ---------- District Pipeline (High Cardinality) ----------
            district_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("freq_encoder", FrequencyEncoder())
            ])

            # ---------- Other Categorical Pipeline ----------
            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "onehot",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False)
                )
            ])

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", num_pipeline, num_col),
                    ("district", district_pipeline, [district_col]),
                    ("cat", cat_pipeline, cat_col)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path: str, test_path: str):
        try:
            logging.info("Starting data transformation")

            # ---------- Read data ----------
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            target_col = "WinningParty"
            district_col = "District"

            # ---------- Identify columns ----------
            num_col = train_df.select_dtypes(include=["int64", "float64"]).columns.tolist()
            cat_col = train_df.select_dtypes(include=["object"]).columns.tolist()

            # Remove target
            if target_col in num_col:
                num_col.remove(target_col)
            if target_col in cat_col:
                cat_col.remove(target_col)

            # Remove district from general categorical list
            if district_col in cat_col:
                cat_col.remove(district_col)

            logging.info(f"Numerical columns: {num_col}")
            logging.info(f"District column: {district_col}")
            logging.info(f"Other categorical columns: {cat_col}")

            # ---------- Split features & target ----------
            X_train = train_df.drop(columns=[target_col])
            y_train = train_df[target_col]

            X_test = test_df.drop(columns=[target_col])
            y_test = test_df[target_col]

            # ---------- Preprocessor ----------
            preprocessing_obj = self.get_data_transformer_object(
                num_col=num_col,
                cat_col=cat_col,
                district_col=district_col
            )

            # ---------- Fit & Transform ----------
            X_train_arr = preprocessing_obj.fit_transform(X_train)
            X_test_arr = preprocessing_obj.transform(X_test)

            # ---------- Combine with target ----------
            train_arr = np.c_[X_train_arr, y_train.to_numpy()]
            test_arr = np.c_[X_test_arr, y_test.to_numpy()]

            # ---------- Save preprocessor ----------
            save_object(
                file_path=self.data_transformation_config.preprocessor_file_path,
                obj=preprocessing_obj
            )

            logging.info("Data transformation completed successfully")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)

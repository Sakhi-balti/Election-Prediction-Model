
import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    artifacts_dir: Path = Path('artifacts')
    preprocessor_obj_file_path = artifacts_dir/'preprocessor.pkl'

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        os.makedirs(self.data_transformation_config.artifacts_dir, exist_ok=True)

    def _build_preprocessing(self,num_col:List[str], cat_col:List[str])->ColumnTransformer:
        try:
          
            num_pipeline = Pipeline(
                steps =[
                    ("imputer", SimpleImputer(strategy ='mean')),
                    ('scaler', StandardScaler())
                ]
            )
            cat_pipeline = Pipeline(
                steps =[
                    ("imputer", SimpleImputer(strategy ='most_frequent')),
                    ('scaler', LabelEncoder())
                ]
            )
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num_pipeline', num_pipeline, num_col),
                    ('cat_pipeline', cat_pipeline, cat_col)
                ],
                remainder='drop'
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path:str, test_path:str)->Tuple[np.ndarray, np.ndarray, Path]:
        try:
          train_df = pd.read_csv(train_path)
          test_df = pd.read_csv(test_path)
        
        # Auto detect numerical & categorical features
          numeric_columns = train_df.select_dtypes(include=["int64", "float64"]).columns.tolist()
          categorical_columns = train_df.select_dtypes(include=["object"]).columns.tolist()
        
        # Remove target from auto-detected lists
          if target_col in numeric_columns:
                numeric_columns.remove(target_col)
          if target_col in categorical_columns:
                categorical_columns.remove(target_col)

          logging.info(f"Numeric Columns: {numeric_columns}")
          logging.info(f"Categorical Columns: {categorical_columns}")

          target_col = 'WinningParty'
        
        # Remove the target column from train  and test dataset
          X_train = train_df.drop(target_col, axis =1)
          y_train = train_df[target_col]

          X_test = test_df.drop(target_col, axis = 1)
          y_test = test_df[target_col]

        
        # call the preprocessor object
          preprocessor_obj =self._build_preprocessing(numeric_columns,categorical_columns) 
        # transform the data
          X_train_transform = preprocessor_obj.fit_transform(X_train)
          X_test_transform = preprocessor_obj.transform(X_test)

        # combine transform data with their target values
          train_arr = np.c_[X_train_transform, y_train.values] 
          test_arr = np.c_[X_test_transform, y_test.values]
        
          return(
              train_arr, test_arr,
               self.data_transformation_config.preprocessor_obj_file_path
                )
        except Exception as e:
         raise CustomException(e, sys)
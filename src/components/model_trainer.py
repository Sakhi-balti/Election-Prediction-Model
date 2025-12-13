import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import numpy as np

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, ComplementNB
from sklearn.ensemble import (RandomForestClassifier,
                                GradientBoostingClassifier,
                                AdaBoostClassifier,
                                ExtraTreesClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

@dataclass
class ModelTrainerConfig:
    artifacts_dir: Path = Path('artifacts')
    model_file_path: Path = artifacts_dir/'model.pkl'
    expected_score: float = 0.5
    overfitting_threshold: float = 0.1

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
   
    def initiate_model_trainer(self, train_array: np.ndarray, test_array: np.ndarray):
        try:
            logging.info("Splitting training and testing input data")
            X_train, y_train = train_array[:,:-1], train_array[:,-1]
            X_test, y_test = test_array[:,:-1], test_array[:,-1]

            models = {
                "DecisionTree": DecisionTreeClassifier(),
                "RandomForest": RandomForestClassifier(),
                "GradientBoosting": GradientBoostingClassifier(),
                "AdaBoost": AdaBoostClassifier(),
                "ExtraTrees": ExtraTreesClassifier(),
                "LogisticRegression": LogisticRegression(max_iter=1000),
                "SVC": SVC(),
                "GaussianNB": GaussianNB()
              }

            best_model_name = None
            best_model_score = 0.0
            best_model = None
            results = {}

            for model_name, model in models.items():
                logging.info(f"Training model: {model_name}")

                metrics = evaluate_model(model, X_train, y_train, X_test, y_test)
                results[model_name] = metrics
                logging.info(f"{model_name} metrics: {metrics}")

                test_score = metrics['test_accuracy']
                train_score = metrics['train_accuracy']
                overfitting_check = abs(train_score - test_score)
                
                if test_score > self.model_trainer_config.expected_score and overfitting_check < self.model_trainer_config.overfitting_threshold:
                    if test_score > best_model_score:
                        best_model_score = test_score
                        best_model_name = model_name
                        best_model = model
                        
            save_object(
                file_path=self.model_trainer_config.model_file_path,
                obj=best_model
            )            
            logging.info(f"Best model: {best_model_name} with score: {best_model_score}")
            return {
                "model_name": best_model_name,
                "model_score": best_model_score,
                "report": results,
                "saved_model_path": self.model_trainer_config.model_file_path
            }
        except Exception as e:
            raise CustomException(e, sys)
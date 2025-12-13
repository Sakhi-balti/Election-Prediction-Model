import sys
import numpy as np
import pandas as pd
import pickle
import dill
from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def save_object(file_path: str, obj: object) -> None:
    try:
        logging.info(f"Saving object to {file_path}")
        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)
        logging.info("Object saved successfully")
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_model(model, X_train, y_train, X_test, y_test) -> dict:
    try:
        logging.info("Evaluating model")
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Accuracy
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)

        # F1, Precision, Recall with zero_division=0 to avoid warnings
        train_f1 = f1_score(y_train, y_train_pred, average='weighted', zero_division=0)
        test_f1 = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)

        train_precision = precision_score(y_train, y_train_pred, average='weighted', zero_division=0)
        test_precision = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)

        train_recall = recall_score(y_train, y_train_pred, average='weighted', zero_division=0)
        test_recall = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)

        metrics = {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'train_f1': train_f1,
            'test_f1': test_f1,
            'train_precision': train_precision,
            'test_precision': test_precision,
            'train_recall': train_recall,
            'test_recall': test_recall
        }

        logging.info("Model evaluation completed")
        return metrics
    except Exception as e:
        raise CustomException(e, sys)


# load_object work of a loading  model,preprocess etc

def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
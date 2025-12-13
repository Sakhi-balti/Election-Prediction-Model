import os 
import sys
from sklearn.model_selection import train_test_split
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from pathlib import Path
import pandas as pd

@dataclass
class DataIngestionConfig:
    artifacts_dir: Path = Path('artifacts')
    train_data_dir: Path = artifacts_dir/"train.csv"
    test_data_dir: Path =  artifacts_dir/"test.csv"
    raw_data_dir: Path =   artifacts_dir/'raw.csv'

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig() 
    def initiate_data_ingestion(self):
        try:
            logging.info('clean data loading.......')
            df = pd.read_csv('data/processed/clean_data.csv')
            
            # making the folder dir
            os.makedirs(self.ingestion_config.artifacts_dir, exist_ok=True)

            # store the all data without splitting
            df.to_csv(self.ingestion_config.raw_data_dir, index = False)
            logging.info('save the data on artifact folder')

            # splite the dataset into train and test set
            train_set, test_set = train_test_split(df, test_size=0.25, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_dir, index = False)
            test_set.to_csv(self.ingestion_config.test_data_dir, index = False)
            
            logging.info("Data ingestion completed successfully")
            return (
                str(self.ingestion_config.train_data_dir),
                str(self.ingestion_config.test_data_dir)
            )

        except Exception as e:
            raise CustomException(e, sys) 






    
    
            
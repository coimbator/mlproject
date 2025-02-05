import os
import sys
from src.exception import CustomException
from src.logger import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transform import DatapreprocessConfig, DataPreprocess

@dataclass
class DataIngestConfig:
    train_data_path : str=os.path.join('artifacts', "train.csv")
    test_data_path : str=os.path.join('artifacts', "test.csv")
    raw_data_path : str=os.path.join('artifacts', "raw.csv")

class DataIngest:
    def __init__(self):
        self.ingestion_config= DataIngestConfig()

    def initiate_Ingestion(self):
        logging.info("Intiating the Data Ingestion")

        try:
            df=pd.read_csv('src\dataset\customer_churn_dataset-testing-master.csv')
            logging.info("  -Read the Dataset as Dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)


            logging.info('  -Train Test Split Initiated')
            train_set, test_set= train_test_split(df, test_size=0.2, random_state=50)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("  -Ingestion is Completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e,sys)

if __name__=="__main__":
    obj=DataIngest()
    train_data, test_data= obj.initiate_Ingestion()

    data_preprocess=DataPreprocess()
    data_preprocess.initiate_data_preprocess(train_data, test_data)
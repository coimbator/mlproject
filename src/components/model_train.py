import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from src.logger import logging
from src.utils import save_object, evaluate_model
from src.exception import CustomException

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config= ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            x_train,y_train,x_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            
            models={
                #"Linear Regression": LinearRegression(),
                "Support Vector Machine": SVC(),
                "K-Nearest Neighbor": KNeighborsClassifier(), 
            }

            model_report:dict=evaluate_model(X_train=x_train, y_train=y_train, X_test=x_test, y_test=y_test, models=models)

            #to get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            #to get best model name from dict
            best_model_name= list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score<0.65:
                raise CustomException("No best model found")
            
            logging.info("Best Model Found on both Train and Test Dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(x_test)
            predicted = (predicted > 0.5).astype(int) if predicted.ndim == 1 else predicted
            score = f1_score(y_test, predicted)

            return score

        except Exception as e:
            raise CustomException(e,sys)
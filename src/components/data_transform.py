import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
'''from sklearn.impute import SimpleImputer  #This is to handle misiing values'''
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler
import category_encoders as ce

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DatapreprocessConfig:
    preprocessor_obj_path= os.path.join('artifacts', "preprocessor.pkl")

class DataPreprocess:
    def __init__(self):
        self.datapreprocess_config=DatapreprocessConfig()
    
    def get_data_for_preprocess(self):
        try:
            numerical_col= ["CustomerID",
                            "Age",
                            "Tenure",
                            "Usage Frequency",
                            "Support Calls",
                            "Payment Delay",
                            "Total Spend",
                            "Last Interaction",
                            'Avg_Spend_Per_Month',
                            "Support_Call_Rate",
                            "Payment_Delay_Rate"]
            categorical_col= ["Gender",
                              "Subscription Type",
                              "Contract Length"]
            num_pipeline1= StandardScaler()     #Works well with Logistic regression, SVM, Neural Network
            #num_pipeline2= MinMaxScaler()   #Works well with tree based models like random forrest, XGBoost
            cat_pipeline= Pipeline(
                steps=[
                    #("Binary_Encoder", ce.BinaryEncoder()),
                    ("One_Hot_Encoder",OneHotEncoder(handle_unknown='ignore')),
                    #("Label_Encoder", LabelEncoder()),
                ]
            )

            logging.info(f"Categorical Columns:{categorical_col}")
            logging.info(f"Numercial Columns:{numerical_col}")

            preprocessor=ColumnTransformer(
                [
                    ("Scaler_pipeline",num_pipeline1, numerical_col),
                    #("MinMax_pipeline",num_pipeline2, numerical_col),
                    ("Categorical_pipeline",cat_pipeline, categorical_col)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_preprocess(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test dataset")

            logging.info("Preprocessing is Started")
            preprocessing_obj=self.get_data_for_preprocess()
            target_column_name ="Churn"
            # Avoid division by zero
            train_df['Tenure'] = train_df['Tenure'].replace(0, 1)
            test_df['Tenure'] = test_df['Tenure'].replace(0, 1)

            # Create new features
            train_df['Avg_Spend_Per_Month'] = train_df['Total Spend'] / train_df['Tenure']
            test_df['Avg_Spend_Per_Month'] = test_df['Total Spend'] / test_df['Tenure']
            train_df['Support_Call_Rate'] = train_df['Support Calls'] / train_df['Tenure']
            test_df['Support_Call_Rate'] = test_df['Support Calls'] / test_df['Tenure']
            train_df['Payment_Delay_Rate'] = train_df['Payment Delay'] / train_df['Tenure']
            test_df['Payment_Delay_Rate'] = test_df['Payment Delay'] / test_df['Tenure']

            input_feature_train=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train=train_df[target_column_name]

            input_feature_test=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test=test_df[target_column_name]

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test)

            train_arr= np.c_[input_feature_train_arr, np.array(target_feature_train)]   # 'np c_' Concatenates column-wise, ensuring the target 
            test_arr= np.c_[input_feature_test_arr, np.array(target_feature_test)]      # variable is added as the last column in train_arr and test_arr
                                                                                        # 'np.array(target_feature_train)' ensures the target variable 
                                                                                        # is in the correct format before concatenation.
            logging.info("Saved preprocessing")

            save_object(
                file_path=self.datapreprocess_config.preprocessor_obj_path,
                obj=preprocessing_obj
            )
            return(
                train_arr,
                test_arr,
                self.datapreprocess_config.preprocessor_obj_path,
            )
        except Exception as e:
            raise CustomException(e,sys)
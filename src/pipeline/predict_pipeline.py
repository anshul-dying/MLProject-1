import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
from sklearn.preprocessing import StandardScaler

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = 'artifacts/model.pkl'
            preprocessor_path = 'artifacts/preprocessor.pkl'
            model = load_object(model_path) 
            preprocessor: StandardScaler = load_object(preprocessor_path)
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(
            self,
            gender:str,
            race_ethnicity: str,
            parent_level_of_education,
            lunch:str,
            test_preparation_course:str,
            reading_score:int,
            writing_score:int
    ):
        self.gender                     = gender
        self.race_ethnicity             = race_ethnicity
        self.parent_level_of_education  = parent_level_of_education
        self.lunch                      = lunch
        self.test_preparation_course    = test_preparation_course
        self.reading_score              = reading_score
        self.writing_score              = writing_score

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'gender': [self.gender],
                'race_ethnicity': [self.race_ethnicity],
                'parent_level_of_education': [self.parent_level_of_education],
                'lunch': [self.lunch],
                'test_preparation_course': [self.test_preparation_course],
                'reading_score': [self.reading_score],
                'writing_score': [self.writing_score]
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)
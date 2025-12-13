import sys
from src.exception import CustomException
from src.utils import load_object
import pandas as pd

class PredictionPipeline:
    def __init__(self):
        pass

#--------------------Predict the Output--------------------------->
    def predict(self, feature):
        try:
           model_path = 'artifacts/model.pkl'
           preprocessor_path = 'artifacts/preprocessor.pkl'
   
           model = load_object(file_path= model_path)
           preprocessor = load_object(file_path=preprocessor_path) 
           feature_scale = preprocessor.transform(feature)
           resulte = model.predict(feature_scale)
           return resulte
        except Exception as e:
            raise CustomException(e,sys)


class CustomData:
    def __init__(
            self,
          District,ConstituencyTitle,   Votes,  TotalValidVotes,   TotalRejectedVotes, TotalVotes, TotalRegisteredVoters,Turnout,Year
    ):
        self.District = District
        self.ConstituencyTitle = ConstituencyTitle
        self.Votes = Votes
        self.TotalValidVotes = TotalValidVotes
        self.TotalRejectedVotes = TotalRejectedVotes
        self.TotalVotes = TotalVotes
        self.TotalRegisteredVoters = TotalRegisteredVoters
        self.Turnout = Turnout
        self.Year = Year

# ------------Get the Data into DataFrame-------------------------------------->
    def get_data_frame(self):
        try:
            custom_datafram ={
            "District"  :self.District,
            "ConstituencyTitle": self.ConstituencyTitle,
            "Votes": self.Votes,         
            "TotalValidVotes": self.TotalValidVotes,
            "TotalRejectedVotes": self.TotalRejectedVotes,     
            "TotalRegisteredVoters": self.TotalRegisteredVoters,
            "TotalVotes": self.TotalVotes,
            "Turnout": self.Turnout,
            "Year": self.Year
            }
            return pd.DataFrame(custom_datafram, index=[0])
        except Exception as e:
            raise CustomException(e, sys)
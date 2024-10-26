import logging
from abc import ABC, abstractmethod
import pandas as pd
from sklearn.linear_model import LinearRegression

class Model(ABC):
    """
    Abstract class defining all Models
    """

    @abstractmethod
    def train(self,X_train:pd.DataFrame, y_train:pd.DataFrame) -> None:
        """
        Trains the model
        Args:
            X_train: train data
            y_train: train labels
        Returns:
            None
        """
        pass

class LinearRegressionModel(Model):
    """
    Linear Regression Model
    """
    def train(self,X_train,y_train,**kwargs):
        """
        Trains the model
        Args:
            X_train: Training Data
            y_train: Training Labels
        Returns:
            None
        """
        try:
            reg = LinearRegression(**kwargs)
            reg.fit(X_train,y_train)
            logging.info("Model training Completed")
            return reg
        except Exception as e:
            logging.error("Error in training model: {}".format(e))
            raise e
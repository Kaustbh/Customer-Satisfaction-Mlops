import logging
from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

class Evaluation(ABC):
    """
    Abstract class defining strategy for evaluation our models
    """

    @abstractmethod
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """
        Calculates the scores for the model

        Args:
            y_true: true values
            y_pred: predicted values
        Returns:
            None
        """
        pass

class MSE(Evaluation):
    """
    Evaluation strategy that uses Mean Squared Error
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        try:
            logging.info("Calculating MSE")
            mse = mean_squared_error(y_true, y_pred)
            logging.info("MSE:{}".format(mse))
            return mse
        except Exception as e:
            logging.error("Error in calculating MSE: {}".format(e))
            raise e

class R2(Evaluation):
    """
    Evaluation Strategy that uses Root Mean Squared Error
    """

    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        try:
            logging.info("Calculating R2 Score")
            r2 = r2_score(y_true, y_pred)
            logging.info("R2:{}".format(r2))
            return r2
        except Exception as e:
            logging.error("Error in calculating R2 score: {}".format(e))
            raise e
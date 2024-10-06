import logging
import pandas as pd
from zenml import step
from src.evaluation import R2, MSE
from sklearn.base import RegressorMixin
import numpy as np
from typing import Tuple
from typing_extensions import Annotated
import mlflow
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def evaluate_model(model: RegressorMixin,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame

) -> Tuple[
    Annotated[float, "R2"],
    Annotated[float, "Root Mean Squared Error"],
]:
    """
    Evaluates the model on the ingested data.

    Args:
        model: Trained model
        X_test: pd.DataFrame,
        y_test: pd.DataFrame
    Returns:
        R2: R2 score
        Root Mean Squared Error: RMSE
    """
    try:
        prediction = model.predict(X_test)
        mse_class = MSE()
        mse = mse_class.calculate_scores(y_test, prediction)
        mlflow.log_metric("MSE", mse)
        r2_class = R2()
        r2 = r2_class.calculate_scores(y_test, prediction)
        mlflow.log_metric("R2", r2)
        logging.info("R2:{}".format(r2))
        logging.info("MSE:{}".format(mse))

        return r2, np.sqrt(mse)

    except Exception as e:
        logging.error("Error in evaluating model: {}".format(e))
        raise e
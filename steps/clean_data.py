import logging
import pandas as pd
from zenml import step
from src.data_cleaning import DataCleaning, DataPreProcessing, DataDivideStrategy
from typing_extensions import Annotated
from typing import Tuple
@step
def clean_data(df:pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"],
]:
    """
    Cleans the data and divides it into train and test

    Args:
        df: input data
    Returns:
        data: train and test data
    """
    try:
        process_strategy = DataPreProcessing()
        data_cleaning = DataCleaning(df,process_strategy)
        processed_data = data_cleaning.handle_data()

        divide_strategy = DataDivideStrategy()
        data_cleaning = DataCleaning(processed_data, divide_strategy)
        X_train, X_test, y_train, y_test = data_cleaning.handle_data()
        logging.info("Data cleaning completed successfully")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error("Error in data cleaning: {}".format(e))
        raise e
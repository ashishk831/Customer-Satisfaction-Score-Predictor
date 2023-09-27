import logging

import mlflow
from zenml import step
import pandas as pd
from sklearn.base import RegressorMixin
from typing_extensions import Annotated
from typing import Tuple
from zenml.client import Client

from src.evaluation import MSE,R2,RMSE

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def evaluate_model(model: RegressorMixin,
                   X_test: pd.DataFrame,
                   y_test: pd.DataFrame,
                   ) -> Tuple[
                       Annotated[float, "r2_score"],
                       Annotated[float, "rmse"]
                   ]:
    """
    Evaluates the model on the ingested data.
    
    Args:
        model: RegressorMixin
        x_test: pd.DataFrame
        y_test: pd.DataFrame
    
    Returns:
        r2 r2 score,
        rmse RSME
    """
    try:
        prediction = model.predict(X_test)
        mse_class = MSE()
        mse = mse_class.calculate_scores(y_test,prediction)
        mlflow.log_metric("mse",mse)
        
        r2_class = R2()
        r2 = r2_class.calculate_scores(y_test,prediction)
        mlflow.log_metric("r2",r2)

        rmse_class = RMSE()
        rmse = rmse_class.calculate_scores(y_test,prediction)
        mlflow.log_metric("rmse",rmse)
        return r2,rmse
    except Exception as e:
        logging.error("Error in evaluating model: {}".format(e))
        raise e
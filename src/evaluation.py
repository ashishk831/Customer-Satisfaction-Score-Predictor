import logging
from abc import ABC, abstractmethod

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

class Evaluation(ABC):
    """
    Abstract class defining strategy for evaluation our models
    """

    @abstractmethod
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate the scores for the model
        Args:
            y_trains: True labels
            y_pred" Predicted labels
        Returns:
            None
        """
        pass

class MSE(Evaluation):
    """
    Evaluation strategy that uses Mean squared error
    Args:
            y_true: np.ndarray
            y_pred: np.ndarray
        Returns:
            mse: float
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        try:
            logging.info("Calculating MSE")
            mse = mean_squared_error(y_true, y_pred)
            logging.info("MSE: {}".format(mse))
            return mse
        except Exception as e:
            logging.error("Error in calculating MSE: {}".format(e))
            raise e

class R2(Evaluation):
    """
    Evaluation strategy that uses R2 score
    Args:
            y_true: np.ndarray
            y_pred: np.ndarray
        Returns:
            mse: float
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        try:
            logging.info("Calculating R2 score")
            r2 = r2_score(y_true, y_pred)
            logging.info("R2 score: {}".format(r2))
            return r2
        except Exception as e:
            logging.error("Error in calculating R2 score: {}".format(e))
            raise e
        
class RMSE(Evaluation):
    """
    Evaluation strategy that uses class RMSE
    Args:
            y_true: np.ndarray
            y_pred: np.ndarray
        Returns:
            mse: float
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray)  -> float:
        try:
            logging.info("Calculating RMSE")
            rmse = mean_squared_error(y_true, y_pred, squared=False)
            logging.info("RMSE: {}".format(rmse))
            return rmse
        except Exception as e:
            logging.error("Error in calculating RMSE: {}".format(e))
            raise e
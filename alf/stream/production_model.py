import logging
from abc import ABC, abstractmethod

import joblib
import numpy as np


class ProductionModel():
    """
    Abstract class for model deployed in production. Model is trained in poolAL process so in this
    class it happans only prediction. 
    """
    def __init__(self, model_path, features_list) -> None:
        self._model_path = model_path
        self._features_list= features_list
        self._clf = None


    def reload(self) -> None:
        try:
            self._clf = joblib.load(self._model_path)
            logging.info(f"Model {self._model_path} LOADED.")
        except FileNotFoundError:
            logging.info(f"Model {self._model_path} not found. No change applied.")

    
    def predict(self, to_predict) -> np.ndarray:
        """
        Predict labels.

        Args:
            to_predict (ip_flow.IPFlowsDataFrame): flows to predict

        Returns:
            np.ndarray: prediction P(c|x) for all classes

        """
        return self._clf.predict_proba(to_predict[self._features_list])

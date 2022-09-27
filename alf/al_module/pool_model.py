import logging
from abc import ABC, abstractmethod

import joblib
import numpy as np
import pandas as pd


class MLModel(ABC):
    """Abstract class for machine learning model or models.
    """
    def __init__(self, ml_model, model_path, experiment_id) -> None:
        """Initialize machine learning model with given model.

        Args:
            ml_model (sklearn model): ML model which meets sklearn API
        """
        self._model_path = model_path
        self._experiment_id = experiment_id
        try:
            self.unpickle()
        except (FileNotFoundError, ValueError):
            self._clf = ml_model

    @abstractmethod
    def train(self) -> None:
        """Train machine learning model.
        """

    @abstractmethod
    def predict(self, to_predict: pd.DataFrame) -> np.ndarray:
        """Predict labels for given data.

        Args:
            to_predict (ip_flow.IPFlowsDataFrame): flows to predict

        Returns:
            np.ndarray: predict probabilities P(c|x) for all classes
        """

    def unpickle(self) -> None:
        """Load classifier from file.
        """
        self._clf = joblib.load(f"{self._model_path}/classifier.{self._experiment_id}.bin")

    def pickle(self) -> None:
        """Save classifier to file.
        """
        joblib.dump(self._clf, f"{self._model_path}/classifier.{self._experiment_id}.bin")


class SupervisedMLModel(MLModel):
    """Class where supervised learning is implemented. This implementation
    is basically adapter class for sklearn models.
    """
    def classes(self) -> np.ndarray:
        return self._clf.classes_

    def predict(self, to_predict: pd.DataFrame, features: list) -> np.ndarray:
        """Predict class based on ML model, using predict_proba function.

        Args:
            to_predict (pd.DataFrame): DataFrame with anotated data to predict.

        Returns:
            ndarray: Same as predict_proba, (n_samples, n_classes).
        """
        return self._clf.predict_proba(to_predict[features])

    def train(self, trainset: pd.DataFrame, features: list) -> None:
        """Train ML model.
        """
        logging.info("Train start.")
        X = trainset.drop(columns=['class'])
        y = trainset['class']

        self._clf.fit(X[features], y)
        logging.info("Train finished.")
        self.pickle()


class CommitteeMLModel(SupervisedMLModel):
    """ Implements committee learning algorithm. Uses VotingClassifier from
    sklearn. Only difference from SupervisedMLModel is output of predict
    method.
    """
    def predict(self, to_predict: pd.DataFrame, features: list) -> np.ndarray:
        """Predict class based on ML model, using predict_proba function.

        Args:
            to_predict (pd.DataFrame): DataFrame with anotated data to predict.

        Returns:
            ndarray: (n_samples, n_models, n_classes)
        """
        decisions = []
        for member in self._clf.estimators_:
            decisions.append(
                member.predict_proba(to_predict[features])
            )
        # transpose because it is much useful to iterate over rows than
        # iterate over model decisions
        return np.array(decisions).transpose(1, 0, 2).astype(float)

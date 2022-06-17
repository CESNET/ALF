import logging
from abc import ABC, abstractmethod

import joblib
import numpy as np

from . import context_manager, d_manager, ip_flow


class MLModel(ABC):
    """Abstract class for machine learning model or models.
    """
    def __init__(self, ml_model) -> None:
        """Initialize machine learning model with given model.

        Args:
            ml_model (sklearn model): ML model which meets sklearn API
        """
        try:
            self.unpickle()
        except (FileNotFoundError, ValueError):
            self._clf = ml_model

    @abstractmethod
    def train(self) -> None:
        """Train machine learning model.
        """

    @abstractmethod
    def predict(self, to_predict: ip_flow.IPFlowsDataFrame) -> np.ndarray:
        """Predict labels for given data.

        Args:
            to_predict (ip_flow.IPFlowsDataFrame): flows to predict

        Returns:
            np.ndarray: predict probabilities P(c|x) for all classes
        """

    @abstractmethod
    def predict_hard(self, to_predict: ip_flow.IPFlowsDataFrame):
        """Make an prediction but output is not a posteriori probability but
        hard decision.
        """

    @abstractmethod
    def classes(self) -> np.ndarray:
        """Get list of classes.
        """

    def unpickle(self) -> None:
        """Load classifier from file.
        """
        ctx = context_manager.ContextProvider.get_context()
        wd = ctx.get_working_dir()
        exp_id = ctx.get_experiment_id()
        self._clf = joblib.load(f"{wd}/classifier.{exp_id}.bin")

    def pickle(self) -> None:
        """Save classifier to file.
        """
        ctx = context_manager.ContextProvider.get_context()
        wd = ctx.get_working_dir()
        exp_id = ctx.get_experiment_id()
        joblib.dump(self._clf, f"{wd}/classifier.{exp_id}.bin")


class SupervisedMLModel(MLModel):
    """Class where supervised learning is implemented. This implementation
    is basically adapter class for sklearn models.
    """
    def classes(self) -> np.ndarray:
        return self._clf.classes_

    def predict(self, to_predict: ip_flow.IPFlowsDataFrame) -> np.ndarray:
        """Predict class based on ML model, using predict_proba function.

        Args:
            to_predict (pd.DataFrame): DataFrame with anotated data to predict.

        Returns:
            ndarray: Same as predict_proba, (n_samples, n_classes).
        """
        features = context_manager.ContextProvider.get_context().get_features()
        return self._clf.predict_proba(to_predict[features])

    def predict_hard(self, to_predict: ip_flow.IPFlowsDataFrame):
        """Predict class based on ML model, using predict function.

        Args:
            to_predict (pd.DataFrame): DataFrame with anotated data to predict.

        Returns:
            ndarray: Same as predict, see sklearn documentation.
        """
        features = context_manager.ContextProvider.get_context().get_features()
        return self._clf.predict(to_predict[features])

    def train(self) -> None:
        """Train ML model.
        """
        logging.info("Train start.")
        features = context_manager.ContextProvider.get_context().get_features()
        X, y = d_manager.DbProvider.get_context().get_train_set()
        self._clf.fit(X[features], y)
        logging.info("Train finished.")
        self.pickle()


class SupervisedMLModelIncremental(SupervisedMLModel):
    """Allows use "online" training of ML model. Underlaying model needs
    implement `partial_fit` method.
    """
    def train(self) -> None:
        """Train ML model with trainset database defined in constants.
        """
        logging.info("Train start.")
        features = context_manager.ContextProvider.get_context().get_features()
        X, y = d_manager.DbProvider.get_context().get_train_set()
        self._clf.partial_fit(X[features], y)
        logging.info("Train finished.")
        self.pickle()


class CommitteeMLModel(SupervisedMLModel):
    """ Implements committee learning algorithm. Uses VotingClassifier from
    sklearn. Only difference from SupervisedMLModel is output of predict
    method.
    """
    def predict(self, to_predict: ip_flow.IPFlowsDataFrame) -> np.ndarray:
        """Predict class based on ML model, using predict_proba function.

        Args:
            to_predict (pd.DataFrame): DataFrame with anotated data to predict.

        Returns:
            ndarray: (n_samples, n_models, n_classes)
        """
        features = context_manager.ContextProvider.get_context().get_features()
        decisions = []
        for member in self._clf.estimators_:
            decisions.append(
                member.predict_proba(to_predict[features])
            )
        # transpose because it is much useful to iterate over rows than
        # iterate over model decisions
        return np.array(decisions).transpose(1, 0, 2).astype(float)

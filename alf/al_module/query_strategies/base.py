import random
from abc import ABC, abstractmethod
import sklearn.pipeline

class QueryStrategy(ABC):
    """QueryStrategy class provides interface for query strategies.
    """

    def __init__(
        self,
        oraculum,
        model: sklearn.pipeline.Pipeline
    ) -> None:
        """Initialize query strategy. Set oraculum/anotator and options.
        """
        self._oraculum = oraculum
        self._model = model

    @abstractmethod
    def select(
        self,
        pool
    ) -> list:
        pass
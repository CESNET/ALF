from abc import abstractmethod
import numpy.random as random
from scipy.stats import entropy
import numpy as np

from . import base

class Uncertainty(base.QueryStrategy):
    def __init__(self, oraculum, model) -> None:
        super().__init__(oraculum, model)

    def select(self, pool: list) -> list:
        probabilities = self._model.predict_proba(pool)
        selected = self._calculate_uncert(probabilities)
        return self._oraculum.annotate(
            selected
        )
        
    @abstractmethod
    def _calculate_uncert(self, proba):
        pass


class UncertaintyEntropy(Uncertainty):
    def _calculate_uncert(proba: np.ndarray):
        return np.argmax(np.transpose(entropy(np.transpose(proba))))




        






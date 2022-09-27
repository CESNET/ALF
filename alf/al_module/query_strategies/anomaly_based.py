import sklearn.ensemble
import numpy as np

from . import base


class IsolationForest(base.QueryStrategy):
    def __init__(self, oraculum, model) -> None:
        super().__init__(oraculum, model)

    def select(self, pool: list) -> list:
        iforest = sklearn.ensemble.IsolationForest()
        results: np.ndarray = iforest.fit_predict(pool)
        return results







import numpy.random as random

from . import base

class RandomStrategy(base.QueryStrategy):
    def __init__(self, oraculum, model) -> None:
        super().__init__(oraculum, model)

    def select(self, pool: list) -> list:
        return self._oraculum.annotate(
            pool[random.randint(0,len(pool))]
        )


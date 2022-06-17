from abc import ABC, abstractmethod
import logging
from imblearn.under_sampling import RandomUnderSampler

from . import d_manager
from . import context_manager


class Postprocessor(ABC):
    """Functor for postprocessing of data in db manager. For example, It could
    be used for undersampling or oversampling.
    """

    @abstractmethod
    def postprocess(self) -> None:
        """Postprocess data in db. No return value or input needed, because
        db manager is implemented as singleton in this framework.
        """

    def _commit(self) -> None:
        context_manager.ContextProvider.get_context().commit()
        d_manager.DbProvider.get_context().commit()


class PostprocessorIdentity(Postprocessor):
    """It does nothing but commit metrics and db.
    """

    def postprocess(self) -> None:
        self._commit()


class PostprocessorUndersample(Postprocessor):
    """It does undersampling.
    """
    def __init__(self, maxsize: int) -> None:
        """Initialize undersampler processor with max size of db.
        """
        super().__init__()
        self._maxsize = maxsize

    def postprocess(self) -> None:
        """It does undersampling.
        """
        logging.info("Doing undersampling.")
        ctx = d_manager.DbProvider.get_context()
        flows = ctx.get_all()
        X = flows.drop(columns=['class'])
        y = flows['class']
        ros = RandomUnderSampler()
        X_resampled, y_resampled = ros.fit_resample(X, y)
        flows = X_resampled
        flows['class'] = y_resampled
        if len(flows) > self._maxsize:
            ctx.set_all(flows.sample(self._maxsize))
        else:
            ctx.set_all(flows)
        self._commit()
        logging.info("Finish undersampling.")


class PostprocessorHumanAnotate(Postprocessor):
    """Human anotator.
    """

    def postprocess(self) -> None:
        """It does undersampling.
        """
        ctx = d_manager.DbProvider.get_context()
        flows = ctx.get_all()
        flows["history"] = flows["class"]
        flows["class"] = flows["human"]
        ctx.set_all(flows)
        self._commit()

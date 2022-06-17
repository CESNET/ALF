import datetime

from . import ip_flow
from . import ml_model
from . import query_strategy
from . import evaluator

from . import context_manager
from . import d_manager


class Processor:
    """Class for processing preprocessed data. It runs prediction,
    query and anotation. It also measure the time of run and runs given
    evaluator.
    """
    def __init__(
            self,
            ml_model_obj: ml_model.MLModel,
            query_strategy_obj: query_strategy.QueryStrategy,
            evaluator_obj: evaluator.Evaluator) -> None:
        """Initialize processor which process input. Need anotator,
        input manager, query strategy, model and evaluator.
        """
        self._model = ml_model_obj
        self._query_strategy = query_strategy_obj
        self._evaluator = evaluator_obj

    def process(self, ip_flows: ip_flow.IPFlows) -> None:
        """Process incoming IP flows. Do the process"""

        d_manager.DbProvider.get_context().fetch(test_size=0.3)

        #  train model
        t1 = datetime.datetime.now()
        self._model.train()
        t2 = datetime.datetime.now()
        train_t_d = t2 - t1

        # prediction
        t1 = datetime.datetime.now()
        prediction = self._model.predict(ip_flows)
        t2 = datetime.datetime.now()
        prediction_t_d = t2 - t1

        # query
        t1 = datetime.datetime.now()
        anotated, mask_anotated = self._query_strategy.select(
            class_proba=prediction,
            flows=ip_flows,
            classes=self._model.classes()
        )
        t2 = datetime.datetime.now()
        query_t_d = t2 - t1

        d_manager.DbProvider.get_context().append_to_db(
            anotated.iloc[mask_anotated]
        )

        # evaluate score
        t1 = datetime.datetime.now()
        self._evaluator.evaluate(
            self._model.classes(), self._model,
            prediction, anotated, mask_anotated)
        t2 = datetime.datetime.now()
        evaluation_t_d = t2 - t1

        context_manager.ContextProvider.get_context().append_metrics({
            "prediction_t": prediction_t_d.total_seconds(),
            "query_t": query_t_d.total_seconds(),
            "evaluation_t": evaluation_t_d.total_seconds(),
            "train_t": train_t_d.total_seconds()
        })

import logging

from . import input_manager
from . import ml_model
from . import evaluator
from . import query_strategy
from . import processor
from . import preprocess
from . import postprocess
from . import context_manager


class Engine():
    """Engine of the framework. Runs preprocessor, processor, postprocessor.
    """
    def __init__(
            self,
            preprocessor: preprocess.Preprocessor,
            postprocessor: postprocess.Postprocessor,
            input_manager_obj: input_manager.InputManager,
            ml_model_obj: ml_model.MLModel,
            evaluator_obj: evaluator.Evaluator,
            query_strategy_obj: query_strategy.QueryStrategy) -> None:
        """Initialize processor which process input.
        Need input manager, query strategy, model and evaluator,
        preprocessor and postprocessor. Most of the classes and methods are
        implemented. User needs implement anotator which is very specific
        part to have default setting.

        It implements base loop in terms of active learning meta-algorithm.

        1. It loads data from input manager.
        2. Preprocess data.
        3. Process data (predict, query, anotate, evaluate).
        4. Save & commit and go to 1.
        """
        self._model = ml_model_obj
        self._input_manager = input_manager_obj
        self._evaluator = evaluator_obj
        self._query_strategy = query_strategy_obj
        self._processor = processor.Processor(
            ml_model_obj=self._model,
            query_strategy_obj=self._query_strategy,
            evaluator_obj=self._evaluator)
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor

    def run(self) -> None:
        """Run command.
        """
        logging.info("Running engine. Start now")
        i = 0
        for flows in self._input_manager.get():
            logging.info("Generation %s", i)
            ctx = context_manager.ContextProvider.get_context()
            ctx.append_metrics({"flows_start": len(flows)})
            flows = self._preprocessor.preprocess(flows)
            self._processor.process(flows)
            ctx.append_metrics({"flows_processed": len(flows)})
            self._postprocessor.postprocess()
            i += 1

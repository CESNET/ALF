from abc import ABC, abstractmethod
import numpy as np
from sklearn import metrics

from alf import ip_flow

from . import d_manager
from . import context_manager
from . import ml_model


class Evaluator(ABC):
    """Interface for Evaluator functor. It gives a score of performance.
    """
    @abstractmethod
    def evaluate(
        self,
            classes: np.ndarray,
            model: ml_model.MLModel,
            prediction: np.ndarray,
            anotate: ip_flow.IPFlows,
            mask_anotate: np.ndarray,
            **args) -> bool:
        """Perfromes a evaluation.

        Args:
            classes (np.ndarray): List of classes
            model (ml_model.MLModel)
            prediction (np.ndarray): prediction done by model
            anotate (ip_flow.IPFlows): anotated flows
            mask_anotate (np.ndarray): indices or mask of flows which are
            anotated

        Returns:
            bool: _description_
        """

    def _send_report(self, metrics_name, y_pred, y_true) -> None:
        """Send report to context.

        Args:
            metrics_name (str): name of metrics
            y_pred: predicted labels
            y_true: true labels
        """
        context_manager.ContextProvider.get_context().append_metrics({
            metrics_name: {
                'accuracy': metrics.accuracy_score(
                    y_true, y_pred),
                'precision': metrics.precision_score(
                    y_true, y_pred, average='macro', zero_division=0),
                'recall': metrics.recall_score(
                    y_true, y_pred, average='macro', zero_division=0),
                'f1': metrics.f1_score(
                    y_true, y_pred, average='macro', zero_division=0),
                "mcc": metrics.matthews_corrcoef(
                    y_true, y_pred),
                "prec_prev": self._precision_from_tpr_fpr(y_true, y_pred)
            }
        })

    def _true_positive_rate(self, y_true, y_pred):
        t_p = ((y_pred == 1) & (y_true == 1)).sum()
        f_n = ((y_pred == 0) & (y_true == 1)).sum()
        return t_p / (t_p + f_n)

    def _false_positive_rate(self, y_true, y_pred):
        f_p = ((y_pred == 1) & (y_true == 0)).sum()
        t_n = ((y_pred == 0) & (y_true == 0)).sum()
        return f_p / (f_p + t_n)

    def _precision_from_tpr_fpr(self, y_true, y_pred):
        """Calculate precision from true positive rate and false positive rate.

        https://github.com/CiscoCTA/nci_eval
        """
        db_ctx = d_manager.DbProvider.get_context()
        db = db_ctx.get_all()
        prevalence = len(db[db["class"] == 1]) / len(db)
        return \
            (prevalence * self._true_positive_rate(y_true, y_pred)) \
            / (prevalence * self._true_positive_rate(y_true, y_pred)
                + ((1 - prevalence)
                    * self._false_positive_rate(y_true, y_pred)))


class EvaluatorTestAndAnotated(Evaluator):
    """Evaluate performance of the model on test data and newly annotated data
    (aka last added to db).
    """
    def evaluate(
        self,
            classes,
            model: ml_model.MLModel,
            prediction: np.ndarray,
            anotate: ip_flow.IPFlowsDataFrame,
            mask_anotate: np.ndarray,
            **args) -> bool:
        if prediction.ndim == 3:
            prediction = prediction.mean(axis=1)
        predicts = np.array([classes[np.argmax(x)] for x in prediction])
        ctx = d_manager.DbProvider.get_context()
        X_test, y_test = ctx.get_test_set()
        y_predict = model.predict_hard(X_test)
        self._send_report("test_set", y_predict, y_test)
        self._send_report(
            "new anotated", predicts[mask_anotate], anotate["class"])
        return True


class EvaluatorTestAnotatedAndAllPredicted(Evaluator):
    """Evaluate test set, newly anotated set and all predicted flows. It needs
    query strategies by in `dry run` mode so they anotate everything.
    This should be used for experiments and theorethical measuring only.
    In real traffic it could cause anotator overflow.
    """
    def evaluate(
        self,
            classes,
            model: ml_model.MLModel,
            prediction: np.ndarray,
            anotate: ip_flow.IPFlowsDataFrame,
            mask_anotate: np.ndarray,
            **args) -> bool:
        if prediction.ndim == 3:
            prediction = prediction.mean(axis=1)
        predicts = np.array([classes[np.argmax(x)] for x in prediction])
        ctx = d_manager.DbProvider.get_context()
        X_test, y_test = ctx.get_test_set()
        y_predict = model.predict_hard(X_test)
        self._send_report("test_testset", y_predict, y_test)
        self._send_report(
            "test_only_anotated",
            predicts[mask_anotate], anotate.iloc[mask_anotate]["class"])
        self._send_report(
            "test_all_predicted", predicts, anotate["class"]
        )
        return True

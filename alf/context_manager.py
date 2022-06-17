import datetime
import json
import logging
import os
from abc import ABC, abstractmethod

import pymysql

from . import provider


class Context(ABC):
    """Base class of Context."""
    def __init__(self) -> None:
        """
        Initialize the context.
        """
        self._metrics = {}
        self._id = None
        self._wd = None
        self._features = None

    def set_experiment_id(self, exp_id: str) -> None:
        """Set unique experiment id.

        Args:
            exp_id (str): Unique experiment id

        Exception:
            TypeError: If exp_id is not a string
        """
        if not isinstance(exp_id, str):
            raise TypeError("Experiment id must be a string")
        self._id = exp_id
        return

    def get_experiment_id(self) -> str:
        """Get experiment id.

        Returns:
            str: Experiment id
        """
        return self._id

    def set_working_dir(self, wd: str) -> None:
        """Set working directory for auxiliary files like db or model backups.

        Args:
            wd (str): Working directory path

        Exception:
            ValueError: If wd is not a directory
            TypeError: If wd is not a string, bytes, os.PathLike or integer

        """
        if not os.path.isdir(wd):
            raise ValueError("Working directory must be a directory")
        self._wd = wd
        return

    def get_working_dir(self) -> str:
        """Get working directory.

        Returns:
            str: Working directory path
        """
        return self._wd

    def set_features(self, features: list[str]) -> None:
        """Set features used by machine learning models.

        Args:
            features (list[str]): List of strings with features names

        Exception:
            ValueError: If features is not a list of strings
        """
        if not isinstance(features, list):
            raise ValueError("Features must be a list of strings")
        for feature in features:
            if not isinstance(feature, str):
                raise ValueError("Features must be a list of strings")
        self._features = features
        return

    def get_features(self) -> list[str]:
        """Get features used by machine learning models.

        Returns:
            list[str]: List of strings with features names
        """
        return self._features

    def append_metrics(self, metric: dict) -> None:
        """Append metrics to the context. Metrics are used to store
        results of experiments as well as performance metrics. In case of
        conflicting names of metrics, the last one is kept. Implemented as
        dictionary.

        Args:
            metric (dict): Dictionary with metrics

        Exception:
            TypeError: If metric is not a dictionary
        """
        if not isinstance(metric, dict):
            raise TypeError("Metric must be a dictionary")
        self._metrics |= metric
        return

    def get_metrics(self) -> dict:
        """Get metrics.

        Returns:
            dict: Dictionary with metrics
        """
        return self._metrics

    @abstractmethod
    def commit(self) -> None:
        """Flush metrics. This method is called when we want save metrics
        dictionary (to STDOUT, file, database, Prometheus etc).
        """


class ContextSQLMetrics(Context):
    """Child class of Context. It is used to store metrics of experiments in
    MySQL database.
    """
    def commit(self) -> None:
        """Flush metrics to MySQL database with following schema columns:

            ``(experiment_id, timestamp, metrics_json)``

        """
        db = pymysql.connect(  # nosec
            host="grafana.liberouter.org",
            user="grafana",
            password="grafana",
            database="grafanadata"
        )
        cur = db.cursor()
        now = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        metrics_json = json.dumps(self._metrics).replace("NaN", "null")
        print(metrics_json)
        cur.execute(f"\
            INSERT INTO `grafanadata`.`alf_experiment_record_v2`\
            (`experiment_id`,`timestamp`,`log`) \
            VALUES(\
                '{self._id}','{now}','{metrics_json}');")
        db.commit()
        cur.close()
        db.close()


class ContextFileMetrics(Context):
    """Child class of Context. It is used to store metrics of experiments in
    file.
    """
    def commit(self) -> None:
        """Flush metrics to file.
        """
        metrics_json = json.dumps(self._metrics)
        logging.info("Metrics dump:\n %s", metrics_json)
        with open(
            f"{self._wd}/metrics_{self._id}.json", "a", encoding="utf8"
        ) as file:
            file.write(f"{metrics_json}\n")
        return


class ContextProvider(provider.Provider):
    """Factory class for Context. It is used to create Context object.
    """
    _ctx = None

    @staticmethod
    def create_context(context_type: str, **options) -> None:
        """Create Context object.

        Args:
            context_type (str): Context type. It can be either
                ``sql`` or ``file``.

        Returns:
            Context: Context object
        """
        if context_type == "sql":
            ContextProvider._ctx = ContextSQLMetrics()
        elif context_type == "file":
            ContextProvider._ctx = ContextFileMetrics()
        else:
            raise ValueError(f"Unknown context type: {context_type}")

    @staticmethod
    def get_context() -> Context:
        """Get context object.

        Returns:
            Context: Context object
        """
        if ContextProvider._ctx is None:
            raise ValueError("Context is not set")
        return ContextProvider._ctx

from abc import ABC, abstractmethod
import pandas as pd
from sklearn.model_selection import train_test_split

from . import ip_flow
from . import provider
from . import context_manager

flow = ip_flow.IPFlowsDataFrame


class DManager(ABC):
    """
    Interface for D_i database
    """
    _db = None

    @abstractmethod
    def get_train_set(self) -> tuple[ip_flow.IPFlows, ip_flow.IPFlows]:
        """
        Get training set
        """

    @abstractmethod
    def get_test_set(self) -> tuple[ip_flow.IPFlows, ip_flow.IPFlows]:
        """
        Get test set
        """

    @abstractmethod
    def get_last_added(self) -> ip_flow.IPFlows:
        """
        Get last added flows
        """

    @abstractmethod
    def append_to_db(self, flows: ip_flow.IPFlows) -> None:
        """Append to database

        Args:
            flows (ip_flow.IPFlows): List of flows to append
        """

    @abstractmethod
    def fetch(self, **options) -> None:
        """
        Load train/test database from engine or file into RAM or cache.
        Internal preprocessing database. Could be used for example divide
        database into train and test set.
        """

    @abstractmethod
    def get_all(self) -> ip_flow.IPFlows:
        """
        Get all flows from database
        """

    @abstractmethod
    def set_all(self, flows: ip_flow.IPFlows) -> None:
        """
        Set all flows in database

        Args:
            flows (ip_flow.IPFlows): List of flows to set
        """

    @abstractmethod
    def commit(self) -> None:
        """Commit changes to database engine or file,
        and cancel train and test set if any

        Usually should be called after append_to_db() and in postprocessing.
        """


class DManagerFile(DManager):
    """DManager implementation for file storage in CSV format. Using pandas
    for CSV parsing and writing.
    """
    _train_tuple = None
    _test_tuple = None
    _new_tuple = None

    def __init__(self, d_0_path: str, **options) -> None:
        """Initialize DManagerFile. During initialization, it loads database
        from CSV file and copy it to working directory. If database in working
        directory exists, nothing is done.

        Args:
            d_0_path (str): Path to file with D_0 database (starting set)
        """
        ctx = context_manager.ContextProvider.get_context()
        wd = ctx.get_working_dir()
        exp_id = ctx.get_experiment_id()

        self._db_path = f"{wd}/db.{exp_id}.csv"
        try:
            pd.read_csv(self._db_path)
            return
        except FileNotFoundError:
            db = pd.read_csv(d_0_path)
            db.to_csv(self._db_path, index=False)
        return

    def fetch(self, **options) -> None:
        """Loads data from CSV file specified by self._db_path. Split into
        train and test sets using train_test_split function from sklearn lib.

        Args:
            train_size (float): Size of train set (0.0 - 1.0)
        """
        test_size = options.get("test_size", 0.3)
        self._db = pd.read_csv(self._db_path)
        X = self._db.drop(columns=['class'])
        y = self._db['class']
        X_train, X_test, y_train, y_test = train_test_split(
                                                    X, y, test_size=test_size)
        self._train_tuple = (X_train, y_train)
        self._test_tuple = (X_test, y_test)
        return

    def get_train_set(self) -> tuple[ip_flow.IPFlows, ip_flow.IPFlows]:
        """Get train set

        Returns:
            ip_flow.IPFlows: List of flows
        """
        return self._train_tuple

    def get_test_set(self) -> tuple[ip_flow.IPFlows, ip_flow.IPFlows]:
        """Get test set

        Returns:
            ip_flow.IPFlows: List of flows
        """
        return self._test_tuple

    def append_to_db(self, flows: ip_flow.IPFlows) -> None:
        """Append flows to database. For the success, it needs to be anotated
        which means it has to have "class" column.

        Args:
            flows (ip_flow.IPFlows): Flows to append.
        """
        if not pd.Series(flows["class"]).notnull().all():
            raise ValueError("Flows to append must be all annotated")
        self._db = pd.concat([self._db, flows])
        X = flows.drop(columns=['class'])
        y = flows['class']
        self._new_tuple = (X, y)
        context_manager.ContextProvider.get_context().append_metrics({
            "new_flows": len(flows),
            "d_size": len(self._db)
        })

    def get_last_added(self) -> ip_flow.IPFlows:
        return self._new_tuple

    def get_all(self) -> ip_flow.IPFlows:
        """Get all flows from database

        Returns:
            ip_flow.IPFlows: List of flows
        """
        return ip_flow.IPFlowsDataFrame(self._db)

    def commit(self) -> None:
        """Commit changes to database file.
        """
        self._db.to_csv(self._db_path, index=False)
        return

    def set_all(self, flows: ip_flow.IPFlows) -> None:
        """Set all flows in database

        Args:
            flows (ip_flow.IPFlows): List of flows to set
        """
        self._db = flows
        return

class DManagerDataFrame(DManagerFile):
    def __init__(self, d_0_path: pd.DataFrame, **options) -> None:
        """Initialize DManagerFile. During initialization, it loads database
        from passed DataFrame object and copy it to working directory. If database in working
        directory exists, nothing is done.

        Args:
            d_0_path (DataFrame): DataFrame pandas table (starting set)
        """
        ctx = context_manager.ContextProvider.get_context()
        wd = ctx.get_working_dir()
        exp_id = ctx.get_experiment_id()

        self._db_path = f"{wd}/db.{exp_id}.csv"
        try:
            pd.read_csv(self._db_path)
            return
        except FileNotFoundError:
            db = d_0_path
            db.to_csv(self._db_path, index=False)
        return
        


class DbProvider(provider.Provider):
    """ Database provider. Implemented as singleton accesible in every part
    od the system.
    """
    _db = None

    @staticmethod
    def create_context(context_type, **options) -> None:
        """Create Di object.

        Args:
            di_database (DManager): Di object

        Returns:
            Di: Di object
        """
        if context_type == "file":
            d_0_path = options.get("d_0_path")
            DbProvider._db = DManagerFile(d_0_path)
        elif context_type == "dataframe":
            d_0_path = options.get("d_0_path")
            DbProvider._db = DManagerDataFrame(d_0_path)
        else:
            raise ValueError("Unknown database type")

    @staticmethod
    def get_context() -> DManager:
        """Get D_i object, a manager for incremental dataset.

        Returns:
            Di: Di object
        """
        if DbProvider._db is None:
            raise ValueError("Db is not set")
        return DbProvider._db

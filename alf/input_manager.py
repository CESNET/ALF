import logging
import os
from abc import ABC, abstractmethod
import pandas as pd

import pytrap

from . import ip_flow


class InputManager(ABC):
    """Abstract class for input manager. It it the input point of the
    framework.
    """
    def __init__(self, definition) -> None:
        """Initialization.
        Args:
            definition (any): Definition of input manager. Depends on
            implementation. Could be name of file, folder, unix socket etc.
        """
        self._input_definition = definition

    @abstractmethod
    def get(self) -> ip_flow.IPFlows:
        """Get all data from defined source as IP flows.

        Returns:
            IPFlows: IP Flows.
        """


class TrapcapFolderInputManager(InputManager):
    """Input manager for trapcaps file. Definition is folder with trapcaps.
    """

    def get(self) -> ip_flow.IPFlowsDataFrame:
        """Get all data from defined source as IP flows."""
        folder = self._input_definition
        for file in os.listdir(folder):
            try:
                path = f"{folder}/{file}"
                nemea_table = pytrap.read_nemea(f"f:{path}", nrows=-1)
                logging.info("Read file %s, #: %s", path, len(nemea_table))
                yield ip_flow.IPFlowsDataFrame(nemea_table)
            except pytrap.TrapError:  # pylint: disable=no-member
                continue
            except UnicodeDecodeError:
                continue
            except AttributeError:
                continue


class TrapcapSocketInputManager(InputManager):
    """Input manager for unix socket. Definition is NEMEA definition.
    """

    def get(self) -> ip_flow.IPFlowsDataFrame:
        """Get all data from defined source as IP flows."""
        socket = self._input_definition
        while True:
            try:
                nemea_table = pytrap.read_nemea(socket, nrows=50000)
                logging.info("Stream from %s, #: %s", socket, len(nemea_table))
                yield ip_flow.IPFlowsDataFrame(nemea_table)
            except pytrap.TrapError:  # pylint: disable=no-member
                continue
            except UnicodeDecodeError:
                continue
            except AttributeError:
                continue


class CSVFolderInputManager(InputManager):
    """Input manager for folder of CSVs. Input is folder
    """

    def get(self) -> ip_flow.IPFlowsDataFrame:
        folder = self._input_definition
        for file in os.listdir(folder):
            try:
                path = f"{folder}/{file}"
                nemea_table = pd.read_csv(path)
                logging.info("Read file %s, #: %s", path, len(nemea_table))
                yield nemea_table
            except pytrap.TrapError:  # pylint: disable=no-member
                continue
            except UnicodeDecodeError:
                continue
            except AttributeError:
                continue

class DataFramesInMemoryInputManager(InputManager):
    """Input manager for in-memory CSV. Input is pandas DataFrame.
    """
    
    def get(self) -> ip_flow.IPFlowsDataFrame:
        for i in self._input_definition:
            yield i


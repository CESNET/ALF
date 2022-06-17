from abc import ABC
import pandas as pd


class IPFlows(ABC):
    """General IP flow class. Needs to be extended by specific underlaying
    data structure. Should follow pandas DataFrame convention for good
    understanding. It is not a good idea to use it directly.
    Idea behind IP Flows class is a table where rows are IP flow and columns
    are ith feature.
    """


class IPFlowsDataFrame(IPFlows, pd.DataFrame):
    """Basically wrapper of pandas DataFrame. Used as abstraction for IP
    flows as it copy its idea."""

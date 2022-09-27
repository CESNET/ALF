from abc import ABC, abstractmethod
import numpy as np

from . import ip_flow


class Anotator(ABC):
    """Class for anotation of given IP flows.
    """
    def __init__(self, **options) -> None:
        """Initialize anotation class. Could pass options as blacklist path
        or whitelist path or HTTP/IP address for requests and so on. It is
        very problem depending.
        """

    @abstractmethod
    def anotate(self, flows: ip_flow.IPFlows) -> ip_flow.IPFlows:
        """Method for anotation.

        Args:
            df (ip_flow.IPFlows): Flows to anotate.

        Returns:
            ip_flow.IPFlows: Anotated data. Basically same as input with
            class extension.
        """


class AnotatorMiners(Anotator):
    """Anotator for Miners. Expects label in flow dataset in "class" column"""

    def anotate(
            self,
            flows: ip_flow.IPFlowsDataFrame) -> ip_flow.IPFlowsDataFrame:
        """Anotate miners. Expects precalculated label in class "column" so
        does nothing.

        Args:
            flows (ip_flow.IPFlowsDataFrame): IP flows

        Returns:
            ip_flow.IPFlowsDataFrame: Anotated IP flows.
        """
        return flows


class AnotatorDoH(Anotator):
    """Anotator for DoH classifiaction problem. Passive anotator base on
    blacklist.
    """
    def __init__(self, blacklist_path: str) -> None:
        """Initialize DOH Anotator class. It is so called passive anotator
        so we just need blacklist with IP list. It could be modified during
        runtime since we only save path.

        Args:
            blacklist_path (str): Path to blacklist file.
        """
        super().__init__()
        self._blacklist = _parse_ip_list(blacklist_path)

    def anotate(
            self,
            flows: ip_flow.IPFlowsDataFrame) -> ip_flow.IPFlowsDataFrame:
        """DOH anotation. Load blacklist and check if any IP in flows DataFrame
        is in blacklist. If so, we mark it as anotated into "class" column.

        Args:
            flows (ip_flow.IPFlowsDataFrame): IP flows

        Returns:
            ip_flow.IPFlowsDataFrame: Anotated IP flows.
        """
        flows_new = flows.copy()
        flows_new["class"] = np.vectorize(self._is_blacklisted)(
            flows['SRC_IP'],
            flows['DST_IP']
        )
        return flows_new

    def _is_blacklisted(self, src_ip, dst_ip):
        ip_blacklist = self._blacklist
        on_blacklist = \
            (str(src_ip) in ip_blacklist) or (str(dst_ip) in ip_blacklist)
        return on_blacklist


def _parse_ip_list(filepath: str) -> set[str]:
    """Parse IP Blacklist from file.

    Args:
        filepath (str): filepath to blacklist file.

    Returns:
        set[str]: set of IPs from blacklist file.
    """
    with open(filepath, mode='r', encoding='utf8') as file:
        return set(file.read().splitlines())

from abc import ABC, abstractmethod
import numpy as np

from pytrap import UnirecTime  # pylint: disable=no-name-in-module

from . import ip_flow


class Preprocessor(ABC):
    """Preprocess class for IP flows. It could extend IP flows with another
    features (feature engineering), normalize values etc.
    """

    @abstractmethod
    def preprocess(self, ip_flows: ip_flow.IPFlows) -> ip_flow.IPFlows:
        """Get preprocessed IP flows.

        Returns:
            IPFlows: IP flows.
        """


class PreprocessorIdentity(Preprocessor):
    """Identity postprocessor. Does nothing."""

    def preprocess(
        self,
            ip_flows: ip_flow.IPFlowsDataFrame) -> ip_flow.IPFlowsDataFrame:
        return ip_flows


class PreprocessorDoH(Preprocessor):
    """Preprocessor for DoH (DNS over HTTPS). It extends flows with statistical
    features. This implementation requires IP Flows in pandas DataFrame format.
    """

    def preprocess(
        self,
            ip_flows: ip_flow.IPFlowsDataFrame) -> ip_flow.IPFlowsDataFrame:
        return self._process(ip_flows)

    def _time2msec(self, timestamp) -> float:
        return timestamp.total_seconds() * 1000 + timestamp.microseconds / 1000

    def _time_diff(self, t1: UnirecTime, t2: UnirecTime) -> float:
        return self._time2msec(t2.toDatetime() - t1.toDatetime())

    def _flow_statistics_time(self, pkt_time):
        if len(pkt_time) < 2:
            return 0, 0, 0, 0, 0, 0
        pkt_time_delays = list(
            map(
                self._time2msec,
                np.diff(
                    list(
                        map(
                            lambda x: x.toDatetime(), pkt_time
                            )
                        )
                    )
                )
        )
        upper_3sigma = np.percentile(pkt_time_delays, 66.3)
        lower_3sigma = np.percentile(pkt_time_delays, 33.4)
        burst = 0
        fizzle = 0
        time_leap_ration = 0
        for delay in pkt_time_delays:
            if delay > upper_3sigma:
                fizzle += 1
            if delay < lower_3sigma:
                burst += 1
        if burst > 0 and fizzle > 0:
            time_leap_ration = burst / fizzle

        if len(pkt_time_delays) > 0:
            mindelay = np.min(pkt_time_delays)
            avgdelay = np.mean(pkt_time_delays)
            maxdelay = np.max(pkt_time_delays)
            return \
                mindelay, avgdelay, maxdelay, burst, fizzle, time_leap_ration
        else:
            return 0, 0, 0, burst, fizzle, time_leap_ration

    def _flow_statistics_packets(self, pkt_directions, pkt_lengths):
        pkt_lens, pkt_lens_rev = [], []
        for pkt_len, pkt_direction in zip(pkt_lengths, pkt_directions):
            if pkt_direction == 1:
                pkt_lens.append(pkt_len)
            elif pkt_direction == -1:
                pkt_lens_rev.append(pkt_len)
        dirLen = len(pkt_directions)
        third = int(dirLen/3)
        stSum = 0
        ndSum = 0
        rdSum = 0
        if third >= 2:
            stSum = np.sum(pkt_directions[:third])
            ndSum = np.sum(pkt_directions[third+1:2*third])
            rdSum = np.sum(pkt_directions[2*third+1:dirLen])
        autocorr = 0
        if len(pkt_lengths) >= 5:
            lags = range(1, min(len(pkt_lengths) // 2, 10))
            np.seterr(divide='ignore', invalid='ignore')
            corr = np.array(
                [np.corrcoef(
                    pkt_lengths[lag:],
                    pkt_lengths[:-lag]
                )[0][1] for lag in lags])
            try:
                lag = np.nanargmax(corr)
            except ValueError:
                lag = None
            if lag:
                autocorr = corr[lag]

        var_pkt_size = np.var(pkt_lens) if len(pkt_lens) > 0 else 0.0
        var_pkt_size_rev = \
            np.var(pkt_lens_rev) if len(pkt_lens_rev) > 0 else 0.0
        median_pkt_size = np.median(pkt_lens) if len(pkt_lens) > 0 else 0.0
        median_pkt_size_rev = \
            np.median(pkt_lens_rev) if len(pkt_lens_rev) > 0 else 0.0
        return \
            var_pkt_size,\
            var_pkt_size_rev,\
            median_pkt_size,\
            median_pkt_size_rev,\
            stSum, ndSum, rdSum, autocorr

    def _process(
            self,
            df: ip_flow.IPFlowsDataFrame) -> ip_flow.IPFlowsDataFrame:
        #  remove one directional
        df.drop(df[df['PACKETS'] == 0].index, inplace=True)
        df.drop(df[df['PACKETS_REV'] == 0].index, inplace=True)

        df['packets_sum'] = df['PACKETS'] + df['PACKETS_REV']

        #  remove short communication
        df.drop(df[df['packets_sum'] < 6].index, inplace=True)

        df['bytes_rev'] = df['BYTES_REV'].astype(int)
        df['bytes'] = df['BYTES'].astype(int)
        df['dst_port'] = df['DST_PORT'].astype(int)
        df['src_port'] = df['SRC_PORT'].astype(int)
        df['packets'] = df['PACKETS'].astype(int)
        df['packets_rev'] = df['PACKETS_REV'].astype(int)
        df['bytes_ration'] = df['bytes'] / df['bytes_rev']
        df['num_pkts_ration'] = df['packets'] / df['packets_rev']
        df['time'] = \
            np.vectorize(self._time_diff)(df['TIME_FIRST'], df['TIME_LAST'])
        df['av_pkt_size'] = df['bytes'] / df['packets']
        df['av_pkt_size_rev'] = df['bytes_rev'] / df['packets_rev']

        var_pkt_size,\
            var_pkt_size_rev,\
            median_pkt_size,\
            median_pkt_size_rev,\
            stSum, ndSum, rdSum, autocorr = np.vectorize(
                self._flow_statistics_packets)(
                    df['PPI_PKT_DIRECTIONS'],
                    df['PPI_PKT_LENGTHS']
                )
        df['var_pkt_size'] = var_pkt_size
        df['var_pkt_size_rev'] = var_pkt_size_rev
        df['median_pkt_size'] = median_pkt_size
        df['median_pkt_size_rev'] = median_pkt_size_rev
        df['stSum'] = stSum
        df['ndSum'] = ndSum
        df['rdSum'] = rdSum
        df['autocorr'] = autocorr

        mindelay,\
            avgdelay, maxdelay, burst, fizzle, time_leap_ration = np.vectorize(
                self._flow_statistics_time)(df['PPI_PKT_TIMES'])
        df['mindelay'] = mindelay
        df['avgdelay'] = avgdelay
        df['maxdelay'] = maxdelay
        df['bursts'] = burst
        df['fizzles'] = fizzle
        df['time_leap_ration'] = time_leap_ration
        return df

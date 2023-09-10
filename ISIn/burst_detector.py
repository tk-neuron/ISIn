import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

from typing import List, Literal, Union


class BurstDetector:
    @classmethod
    def _preprocess(
        cls,
        spiketime: Union[List[float], np.ndarray[float]],
        unit: Literal["s", "ms"]
    ):
        if unit == "s":
            return np.sort(spiketime * 1000.0)
        elif unit == "ms":
            return np.sort(spiketime)
        else:
            raise ValueError("Invalid unit value. unit must be 'ms' or 's'.")

    @classmethod
    def plot(
        cls,
        spiketime: Union[list[float], np.ndarray[float]],
        n_list: List[int],
        ax: plt.Axes = None,
        threshold: float = None,
        spiketime_unit: Literal["s", "ms"] = "s",
        xmin: int = -1,
        xmax: int = 4,
        ymin: int = -7,
        ymax: int = 0,
    ):
        """
        plot ISIn histogram
        :param spiketime_sec: np.array of spike train in sec scale
        :param n_list: list of integer for n (ISI of every nth spike)
        :param threshold_msec: optional, set ISIn threshold to visualize
        """
        spiketime_ms = cls._preprocess(spiketime, spiketime_unit)
        lowess = sm.nonparametric.lowess

        if ax is None:
            fig, ax = plt.subplots()

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("$isi_N [ms]$")
        ax.set_ylabel("probability [%]")
        ax.set_xlim(10 ** xmin, 10 ** xmax)
        ax.set_ylim(10 ** ymin, 10 ** ymax)
        ax.grid()

        for n in n_list:
            isi_n = spiketime_ms[n - 1:] - spiketime_ms[:1 - n]
            hist, edges = np.histogram(isi_n, bins=np.logspace(-1, 4, 100))

            x = edges[1:]
            y = hist / np.sum(hist)
            filtered = lowess(y, x, is_sorted=True, frac=0.1, it=0)
            ax.plot(filtered[:, 0], filtered[:, 1], label="N={}".format(n))

        if threshold is not None:
            ax.vlines(
                x=threshold,
                ymin=10 ** ymin,
                ymax=10 ** ymax,
                colors="red",
                label="threshold",
            )

        ax.legend(loc="lower left")
        return ax

    @classmethod
    def detect(
        cls,
        spiketime: Union[List[float], np.ndarray[float]],
        n: int,
        threshold: float,
        spiketime_unit: Literal["s", "ms"] = "s",
        return_idx: bool = False,
    ):
        spiketime_ms = cls._preprocess(spiketime, spiketime_unit)
        n_spikes = len(spiketime)
        burst_idx = np.zeros(n_spikes, dtype=int)

        for i in range(n_spikes - n + 1):
            if spiketime_ms[i + n - 1] - spiketime_ms[i] <= threshold:
                burst_idx[i:i + n] = True

        if return_idx:
            return burst_idx
        else:
            # extend the train to calculate the difference of burst_idx;
            # so if the very first or last spike of the train is in burst,
            # they can be assigned to burst correctly
            extended_idx = np.append(False, burst_idx)
            extended_idx = np.append(extended_idx, False)
            diff = extended_idx[1:] - extended_idx[:-1]

            burst_start_idx = diff[:-1] == 1
            burst_end_idx = diff[1:] == -1

            burst_start = spiketime[burst_start_idx]
            burst_end = spiketime[burst_end_idx]
            burst = np.vstack([burst_start, burst_end]).T
            return burst

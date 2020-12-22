import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm


class ISIn:
    """
    example usage:
    > nb_detector = ISIn()
    > nb_detector.plot(spiketime_sec=spike_train, n_list=range(2, 10))
    > bursts = nb_detector.burst_detection(spiketime_sec=spike_train, n=10, threshold_msec=50)
    """
    @staticmethod
    def plot(spiketime_sec, n_list, threshold_msec=None):
        """
        plot ISIn histogram
        :param spiketime_sec: np.array of spike train in sec scale
        :param n_list: list of integer for n (ISI of every nth spike)
        :param threshold_msec: optional, set ISIn threshold to visualize
        """
        lowess = sm.nonparametric.lowess
        spiketime_sec = np.sort(spiketime_sec)
        spiketime = spiketime_sec * 1000.0  # convert the scale to ms

        fig, ax = plt.subplots()
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("$isi_N [ms]$")
        ax.set_ylabel("probability [%]")
        ax.set_xlim(10 ** (-1), 10 ** 4)
        ax.set_ylim(10 ** (-7), 10 ** 0)
        ax.grid()

        for i, n in enumerate(n_list):
            isi_n = spiketime[n - 1:] - spiketime[:1 - n]
            hist, edges = np.histogram(isi_n, bins=np.logspace(-1, 4, 100))

            x = edges[1:]
            y = hist / np.sum(hist)
            filtered = lowess(y, x, is_sorted=True, frac=0.1, it=0)  # smooth by lowess
            ax.plot(filtered[:, 0], filtered[:, 1], label='N={}'.format(n))

        if threshold_msec is not None:
            ax.vlines(x=threshold_msec, ymin=10**(-7), ymax=10**0, colors='red', label='threshold')

        ax.legend(loc='lower left')
        plt.show()

    @staticmethod
    def burst_detection(spiketime_sec, n, threshold):
        """
        detect bursts from spike train
        :param spiketime_sec: np.array of spike time in sec scale
        :param n: n for ISIn
        :param threshold: ISIn threshold
        :return: burst array, burst[i] represents ith burst's start time and end time
        """
        spiketime_sec = np.sort(spiketime_sec)
        spiketime_msec = list(spiketime_sec * 1000.0)  # convert the scale to ms
        n_spikes = len(spiketime_msec)
        burst_idx = np.zeros(n_spikes, dtype=np.int)

        for i in range(n_spikes - n + 1):
            if spiketime_msec[i + n - 1] - spiketime_msec[i] <= threshold:
                burst_idx[i:i + n] = True

        # extend the train to calculate the difference of burst_idx;
        # so that even if the very first or last spike of the train is in burst, they can be assigned to burst correctly
        extended_idx = np.append(False, burst_idx)
        extended_idx = np.append(extended_idx, False)
        diff = extended_idx[1:] - extended_idx[:-1]

        burst_start_idx = (diff[:-1] == 1)
        burst_end_idx = (diff[1:] == -1)

        burst_start = spiketime_sec[burst_start_idx]
        burst_end = spiketime_sec[burst_end_idx]
        burst = np.array([burst_start, burst_end]).transpose()

        # burst[i]: ith burst
        # burst[:, 0]: array of all bursts' start time, burst[:, 1]: array of all bursts' end time
        return burst

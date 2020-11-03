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
        :param spiketime_sec: spike train in sec scale
        :param n_list: list of integer for n (ISI of every nth spike)
        :param threshold_msec: optional, set ISIn threshold to visualize
        """
        lowess = sm.nonparametric.lowess
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

    @staticmethod
    def burst_detection(spiketime_sec, n, threshold_msec):
        """
        detect burst based on specified n and isi_n [ms]
        :param spiketime_sec: spike train in sec scale
        :param n: n for ISIn
        :param threshold_msec: ISIn threshold determined in msec scale
        :return: burst array burst[0]: array of burst start time, burst[1]: array of burst end time
        """
        spiketrain = list(spiketime_sec * 1000.0)  # convert the scale to ms
        n_spikes = len(spiketrain)
        burst_start, burst_end = [], []

        burst_on = False

        for i in range(n_spikes - n + 1):
            t_i = spiketrain[i]
            if spiketrain[i + n - 1] - spiketrain[i] <= threshold_msec:
                if not burst_on:
                    burst_start.append(spiketrain[i])
                    burst_on = True

            else:
                if burst_on:
                    burst_end.append(spiketrain[i + n - 2])
                    burst_on = False

        if len(burst_end) == len(burst_start) - 1:
            burst_end.append(t_i)

        burst = np.array([burst_start, burst_end]) / 1000.0
        return burst  # burst[0]: burst_start, burst[1]: burst_end

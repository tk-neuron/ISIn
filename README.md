# Parameters for burst detection (Python)

This code is **unofficial** Python implementation of the following paper's algorithm:

Bakkum DJ, Radivojevic M, Frey U, Franke F, Hierlemann A, Takahashi H. [Parameters for burst detection](https://doi.org/10.3389/fncom.2013.00193). Front Comput Neurosci. 2014 Jan 13;7:193. doi: 10.3389/fncom.2013.00193. PMID: 24567714; PMCID: PMC3915237.



## Libraries

* numpy
* matplotlib
* statsmodels



## Example

You can import `burst_detector.py` as a module.  The algorithm ***ISI_N*** is implemented as a class.

```python
import burst_detector

spike_train = # prepare a single spike train that combines all spikes from multiple channels #
# make sure spike train is in sec scale, not msec scale

nb_detector = burst_detector.ISIn()
nb_detector.plot(spiketime_sec=spike_train, n_list=range(2, 10))
bursts = nb_detector.burst_detection(spiketime_sec=spike_train, n=10, threshold_msec=50)
# note that you specify isi_n threshold in msec scale, not sec scale

print(bursts[i])  # ith burst
print(bursts[:, 0])  # array of all bursts' start time
print(bursts[:, 1])  # array of all bursts' end time
```

Optimal `n_list` varies a lot depending on your data (e.g. culturing density, the number of channels), so I recommend trying various N values.


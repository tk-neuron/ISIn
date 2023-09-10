# Parameters for burst detection (Python3)

This code is **unofficial** Python3 implementation of the following paper's algorithm:

Bakkum DJ, Radivojevic M, Frey U, Franke F, Hierlemann A, Takahashi H. [Parameters for burst detection](https://doi.org/10.3389/fncom.2013.00193). Front Comput Neurosci. 2014 Jan 13;7:193. doi: 10.3389/fncom.2013.00193. PMID: 24567714; PMCID: PMC3915237.

## Install
After cloning this repository, do:
```shell
(venv) cd ISIn
(venv) pip install .  # make sure you're in the virtual environment for your project
```

## Example
See details in examples directory.

```python
from ISIn import BurstDetector

spike_train = np.array([])  # prepare a single spike train (np.array) that combines all spikes from multiple channels
bursts = BurstDetector.detect(spiketime=spike_train, n=500, threshold=100)

print(bursts[i])  # ith burst
print(bursts[:, 0])  # array of all bursts' start time
print(bursts[:, 1])  # array of all bursts' end time
```

Optimal `n_list` varies a lot depending on your data (e.g. culturing density, the number of channels), so I recommend trying various N values.


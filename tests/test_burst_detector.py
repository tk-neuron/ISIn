import numpy as np
import pytest
from ISIn import BurstDetector


def test_preprocess():
    spiketimes = np.array([0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5])
    spiketimes_s = BurstDetector._preprocess(spiketimes, "s")
    spiketimes_ms = BurstDetector._preprocess(spiketimes, "ms")

    assert np.allclose(spiketimes_s, np.array([
        100.0, 500.0, 1000.0, 1500.0, 2000.0,
        2500.0, 3000.0, 3500.0, 4000.0, 4500.0,
    ]))
    assert np.allclose(spiketimes_ms, spiketimes)


if __name__ == "__main__":
    pytest.main()

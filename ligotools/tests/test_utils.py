from pathlib import Path
import numpy as np
import pytest
from ligotools import utils
from scipy.io import wavfile

# whiten test
def test_whiten_basic_properties():
    """
    Verify that utils.whiten outputs a real array of same shape with finite,
    nonzero variance and near-zero mean when given a flat PSD.
    """
    rng = np.random.default_rng(0)
    fs = 1024.0
    dt = 1.0 / fs
    n = 4096
    x = rng.standard_normal(n)

    # Flat PSD = constant value across all freqs
    interp_psd = lambda f: np.full_like(f, 2.0, dtype=float)

    y = utils.whiten(x, interp_psd, dt)

    # shape & dtype checks
    assert isinstance(y, np.ndarray)
    assert y.shape == x.shape
    assert np.isrealobj(y)

    # numeric sanity
    assert np.isfinite(y).all(), "whiten produced non-finite values"

    mean = float(np.mean(y))
    std = float(np.std(y))

    # mean should be close to 0, variance should be positive (not enforcing magnitude)
    assert abs(mean) < 0.05, f"mean too large ({mean})"
    assert std > 0, f"std must be > 0 (got {std})"


# reqshift func test
def test_reqshift_shifts_tone_frequency():
    fs = 2048
    dur = 1.0
    n = int(fs * dur)
    t = np.arange(n) / fs

    f0 = 50.0
    fshift = 20.0
    x = np.sin(2 * np.pi * f0 * t)

    y = utils.reqshift(x, fshift=fshift, sample_rate=fs)

    # Find dominant frequency via FFT peak
    def peak_freq(sig):
        freqs = np.fft.rfftfreq(len(sig), 1.0 / fs)
        spec = np.abs(np.fft.rfft(sig))
        return freqs[np.argmax(spec)]

    pf_x = peak_freq(x)
    pf_y = peak_freq(y)

    # Expect ~ f0 and ~ f0+fshift
    assert abs(pf_x - f0) < 0.5
    assert abs(pf_y - (f0 + fshift)) < 0.75 

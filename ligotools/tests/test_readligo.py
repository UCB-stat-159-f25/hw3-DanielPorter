from pathlib import Path
import json
import numpy as np
import pytest

# Import the package under test
from ligotools import readligo as rl

# --- Constants / paths used in tests
DATA_DIR = Path("data")
EVENTS_JSON = DATA_DIR / "BBH_events_v3.json"
DEFAULT_EVENT = "GW150914"   # any event present in the JSON works


def _skip_if_missing_file(p: Path):
    if not p.exists():
        pytest.skip(f"Missing required file for test: {p} "
                    f"(did you move the .hdf5/.json into data/ as required?)")


@pytest.fixture(scope="module")
def event():
    """Load event metadata once for all tests."""
    _skip_if_missing_file(EVENTS_JSON)
    with open(EVENTS_JSON, "r") as f:
        events = json.load(f)
    assert DEFAULT_EVENT in events, f"{DEFAULT_EVENT} not found in {EVENTS_JSON}"
    return events[DEFAULT_EVENT]


@pytest.mark.parametrize("det", ["H1", "L1"])
def test_loaddata_shapes_and_finiteness(event, det):
    """Basic sanity: loaddata returns finite 1D arrays with matching lengths."""
    # Build the correct path into data/
    key = f"fn_{det}"
    fname = DATA_DIR / event[key]
    _skip_if_missing_file(fname)

    strain, time, chan = rl.loaddata(str(fname), det)

    # Returned structures should be valid
    assert strain is not None and time is not None, "loaddata returned None arrays"
    assert isinstance(chan, dict), "channel dict should be a dict"

    # 1D, same length, non-empty
    strain = np.asarray(strain)
    time = np.asarray(time)

    assert strain.ndim == 1 and time.ndim == 1, "Expected 1D arrays"
    assert len(strain) == len(time) and len(time) > 0, "Length mismatch or empty arrays"

    # Finite values
    assert np.isfinite(strain).all(), "Non-finite values in strain"
    assert np.isfinite(time).all(), "Non-finite values in time"

    # Time should be strictly increasing
    diffs = np.diff(time)
    assert np.all(diffs > 0), "Time vector must be strictly increasing"


@pytest.mark.parametrize("det", ["H1", "L1"])
def test_time_uniform_sampling_matches_fs(event, det):
    """The implied sampling interval from time should match 1/fs from the JSON."""
    key = f"fn_{det}"
    fname = DATA_DIR / event[key]
    _skip_if_missing_file(fname)

    strain, time, _ = rl.loaddata(str(fname), det)
    time = np.asarray(time)
    assert len(time) >= 2, "Time vector too short for sampling check"

    # Use median to be robust to any edge artifacts
    dt = float(np.median(np.diff(time)))
    fs = float(event["fs"])

    # match within a small tolerance
    assert np.isclose(dt, 1.0 / fs, rtol=1e-5, atol=1e-8), f"dt={dt} vs 1/fs={1.0/fs}"


@pytest.mark.parametrize("det", ["H1", "L1"])
def test_channel_dict_has_DATA_flag(event, det):
    """Channel dict should include a DATA mask we can interpret as booleans."""
    key = f"fn_{det}"
    fname = DATA_DIR / event[key]
    _skip_if_missing_file(fname)

    _, time, chan = rl.loaddata(str(fname), det)
    assert isinstance(chan, dict), "Channel info is not a dict"
    assert "DATA" in chan, "Channel dict missing 'DATA' key"

    bits = np.asarray(chan["DATA"]).astype(bool)
    assert bits.ndim == 1 and bits.size > 0, "DATA mask empty or wrong shape"

    # Not all must be True, but mask should be valid booleans
    assert (bits == True).any() or (bits == False).any(), "DATA mask not boolean-like"

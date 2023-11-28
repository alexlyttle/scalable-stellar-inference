import pytest
import numpy as np
from celestify.emulator import Emulator

backends = ["numpy", "jax", "pytensor", "tensorflow"]
alternate_backends = ["np", "pt", "tf"]
inputs = 5 * [0.0]

@pytest.mark.parametrize("backend", backends+alternate_backends)
def test_emulator_init(backend):
    """Test emulator init method works."""
    _ = Emulator(backend=backend)

@pytest.mark.parametrize("backend", backends)
def test_emulator_call(backend):
    """Test emulator call method works."""
    emulator = Emulator(backend=backend)
    _ = emulator(inputs)

def test_emulator_consistency():
    """Test the consistency of the emulator output between backends."""
    first_output = None
    for backend in backends:
        emulator = Emulator(backend=backend)
        output = emulator(inputs)
        if backend == "pytensor":
            output = output.eval()
        if first_output is None:
            first_output = output
        else:
            assert np.allclose(output, first_output)

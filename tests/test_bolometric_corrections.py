import pytest
from celestify.bolometric_corrections import BolometricCorrections

bands = [["BP", "RP", "G"], ["H", "J", "K"]]
inputs = [5772., 4.44, 0.0, 0.0]

@pytest.mark.parametrize("bands", bands)
def test_bc_init(bands):
    _ = BolometricCorrections(bands)

@pytest.mark.parametrize("bands", bands)
def test_bc_call(bands):
    bc = BolometricCorrections(bands)
    _ = bc(inputs)

import pytest
from celestify.star import Star

params = {
    "evol": 0.5,
    "log_mass": 0.0,
    "M_H": 0.0,
    "Y": 0.28,
    "a_MLT": 2.0,
    "distance": 10.0,
    "Av": 0.0,
}
bands = [None, ["BP", "RP", "G"]]

@pytest.mark.parametrize("bands", bands)
def test_star_init(bands):
    """Test star initialization."""
    _ = Star(bands=bands)

@pytest.mark.parametrize("bands", bands)
def test_star_call(bands):
    """Test star callable."""
    star = Star(bands=bands)
    _ = star(params)

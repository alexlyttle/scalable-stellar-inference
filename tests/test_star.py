from celestify.star import Star

params = {
    "evol": 0.5,
    "log_mass": 0.0,
    "M_H": 0.0,
    "Y": 0.28,
    "a_MLT": 2.0,
}

def test_star_init():
    """Test star initialization."""
    _ = Star()

def test_star_call():
    """Test star callable."""
    star = Star()
    _ = star(params)

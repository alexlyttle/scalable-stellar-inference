import pytest
from jax import random
from numpyro import handlers
from celestify.numpyro_models import SingleStarModel

bands = [["BP", "RP", "G"], ["H", "J", "K"]]

@pytest.mark.parametrize("bands", bands)
def test_single_star_init(bands):
    _ = SingleStarModel(bands=bands)

@pytest.mark.parametrize("bands", bands)
def test_single_star_trace(bands):
    model = SingleStarModel(bands=bands)

    rng_seed = random.PRNGKey(0)
    _ = handlers.trace(handlers.seed(model, rng_seed=rng_seed)).get_trace(obs=None)

from pprint import pprint
from jax import random
from numpyro import handlers
from celestify.numpyro_models import SingleStarModel

def test_single_star_init():
    _ = SingleStarModel()

def test_single_star_trace():
    model = SingleStarModel()

    rng_seed = random.PRNGKey(0)
    trace = handlers.trace(handlers.seed(model, rng_seed=rng_seed)).get_trace(obs=None)
    pprint(trace)

import os
import importlib.metadata

__version__ = importlib.metadata.version(__package__)  # only works if package installed via pip
PACKAGEDIR = os.path.abspath(os.path.dirname(__file__))

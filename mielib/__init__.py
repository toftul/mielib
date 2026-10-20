from importlib.metadata import PackageNotFoundError, version

from . import coordinates, mieacoustics, optics, special, vsh
from .mieacoustics import *

__all__ = ["coordinates", "optics", "special", "vsh"] + mieacoustics.__all__

try:
    __version__ = version("mielib")
except PackageNotFoundError:
    # for example, the source folder is on the path but the package is not installed
    __version__ = "unknown"

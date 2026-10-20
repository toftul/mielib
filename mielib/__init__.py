from importlib.metadata import PackageNotFoundError, version

from . import mieacoustics, mieoptics, rotations, special, vsh
from .mieacoustics import *
from .mieoptics import *
from .rotations import *

__all__ = ["special", "vsh"] + mieacoustics.__all__ + mieoptics.__all__ + rotations.__all__

try:
    __version__ = version("mielib")
except PackageNotFoundError:
    # for example, the source folder is on the path but the package is not installed
    __version__ = "unknown"

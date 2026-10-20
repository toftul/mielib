from importlib.metadata import PackageNotFoundError, version

from . import extraspecial, harmonics, mieacoustics, mieoptics, rotations
from .extraspecial import *
from .mieacoustics import *
from .mieoptics import *
from .harmonics import *
from .rotations import *

__all__ = extraspecial.__all__ + mieacoustics.__all__ + mieoptics.__all__ + harmonics.__all__ + rotations.__all__

try:
    __version__ = version("mielib")
except PackageNotFoundError:
    # for example, the source folder is on the path but the package is not installed
    __version__ = "unknown"

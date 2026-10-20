from importlib.metadata import PackageNotFoundError, version

from . import acoustics, coordinates, optics, special, vsh

__all__ = ["acoustics", "coordinates", "optics", "special", "vsh"]

try:
    __version__ = version("mielib")
except PackageNotFoundError:
    # for example, the source folder is on the path but the package is not installed
    __version__ = "unknown"

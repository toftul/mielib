import inspect
import tomllib
from pathlib import Path

import mielib

MODULES = [mielib.special, mielib.vsh, mielib.mieacoustics, mielib.mieoptics, mielib.coordinates]


def test_all_lists_each_public_function_once():
    for module in MODULES:
        defined = [name for name, f in inspect.getmembers(module, inspect.isfunction)
                   if f.__module__ == module.__name__ and not name.startswith('_')]
        assert sorted(module.__all__) == sorted(defined), module.__name__

    assert len(mielib.__all__) == len(set(mielib.__all__))
    assert all(callable(getattr(mielib, name)) or inspect.ismodule(getattr(mielib, name)) for name in mielib.__all__)


def test_namespace_has_no_helper_modules():
    for name in ['np', 'sp']:
        assert not hasattr(mielib, name)


def test_version_matches_pyproject():
    pyproject = tomllib.loads((Path(__file__).parents[1] / 'pyproject.toml').read_text())
    assert mielib.__version__ == pyproject['tool']['poetry']['version']

"""CFACT: physical counterfactual action-tree datasets."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("cfact")
except PackageNotFoundError:
    __version__ = "0.7.1.dev0"

from .data.reader import TreeDataset
from .data.release import ReleaseDataset

__all__ = ["TreeDataset", "ReleaseDataset", "__version__"]

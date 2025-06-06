"""
Package initialization and imports.

This module imports submodules and utility functions for dataset handling.

Imports:
    diting: Module related to the 'diting' dataset or functionality.
    pnw: Module related to the 'pnw' dataset or functionality.
    sos: Module related to the 'sos' dataset or functionality.
    stead: Module related to the 'stead' dataset or functionality.
    build_dataset: Factory function to build dataset instances.
    get_dataset_list: Utility function to retrieve the list of available datasets.
"""

from . import diting, pnw, sos, stead
from ._factory import build_dataset, get_dataset_list

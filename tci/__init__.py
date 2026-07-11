"""Troubled Cell Indicators for discontinuous Galerkin methods.

Core (numpy-only): DG solver, classical/PA indicators, limiters, data generation.
Optional (install extra ``tci[ml]``): GNN indicator, training loop.
"""

from tci.solvers.dg1d import DG1D
from tci.indicators.classical import MinmodIndicator, KXRCFIndicator
from tci.indicators.pa import PAIndicator, polynomial_annihilation

__all__ = [
    "DG1D",
    "MinmodIndicator",
    "KXRCFIndicator",
    "PAIndicator",
    "polynomial_annihilation",
]

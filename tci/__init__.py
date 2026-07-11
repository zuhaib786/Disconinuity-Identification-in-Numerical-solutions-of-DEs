"""Troubled Cell Indicators for discontinuous Galerkin methods.

Core (numpy-only): DG solvers (advection, Burgers, Euler), classical/PA
indicators, limiters, data generation.
Optional (install extra ``tci[ml]``): GNN and MLP indicators, training loop.
"""

from tci.indicators.classical import KXRCFIndicator, MinmodIndicator
from tci.indicators.pa import PAIndicator, polynomial_annihilation
from tci.solvers.burgers import BurgersDG1D
from tci.solvers.dg1d import DG1D
from tci.solvers.euler import EulerDG1D

__all__ = [
    "DG1D",
    "BurgersDG1D",
    "EulerDG1D",
    "MinmodIndicator",
    "KXRCFIndicator",
    "PAIndicator",
    "polynomial_annihilation",
]

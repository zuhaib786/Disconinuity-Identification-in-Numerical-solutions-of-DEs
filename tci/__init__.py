"""Troubled Cell Indicators for discontinuous Galerkin methods.

Core (numpy-only): DG solvers (advection, Burgers, Euler), classical/PA
indicators, limiters, data generation.
Optional (install extra ``tci[ml]``): GNN and MLP indicators, training loop.
"""

from tci.indicators.classical import KXRCFIndicator, MinmodIndicator
from tci.indicators.base import OrIndicator
from tci.indicators.classical2d import MinmodIndicator2D
from tci.indicators.pa import PAIndicator, polynomial_annihilation
from tci.solvers.burgers import BurgersDG1D
from tci.solvers.dg1d import DG1D
from tci.solvers.euler import EulerDG1D
from tci.solvers.dg2d import AdvectionDG2D
from tci.solvers.euler2d import EulerDG2D
from tci.mesh import (
    TriangleMesh,
    delaunay_mesh,
    perturbed_delaunay_mesh,
    forward_step_mesh,
    random_delaunay_mesh,
    rectangular_mesh,
)

__all__ = [
    "DG1D",
    "BurgersDG1D",
    "EulerDG1D",
    "AdvectionDG2D",
    "EulerDG2D",
    "MinmodIndicator",
    "KXRCFIndicator",
    "OrIndicator",
    "MinmodIndicator2D",
    "PAIndicator",
    "polynomial_annihilation",
    "TriangleMesh",
    "rectangular_mesh",
    "delaunay_mesh",
    "random_delaunay_mesh",
    "perturbed_delaunay_mesh",
    "forward_step_mesh",
]

from abc import ABC, abstractmethod

import numpy as np


class Indicator(ABC):
    """A troubled-cell indicator.

    Given a DG solver and the current nodal solution u of shape (Np, K),
    ``flag`` returns a boolean array of shape (K,) marking troubled cells.
    """

    @abstractmethod
    def flag(self, solver, u):
        raise NotImplementedError


class OrIndicator(Indicator):
    """Flags every cell selected by either child indicator."""

    def __init__(self, left, right):
        self.left = left
        self.right = right

    def flag(self, solver, u):
        return np.logical_or(
            self.left.flag(solver, u), self.right.flag(solver, u)
        )

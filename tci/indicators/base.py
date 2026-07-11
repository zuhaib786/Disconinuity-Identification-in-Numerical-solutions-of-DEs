from abc import ABC, abstractmethod


class Indicator(ABC):
    """A troubled-cell indicator.

    Given a DG solver and the current nodal solution u of shape (Np, K),
    ``flag`` returns a boolean array of shape (K,) marking troubled cells.
    """

    @abstractmethod
    def flag(self, solver, u):
        raise NotImplementedError

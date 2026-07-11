from tci.indicators.base import Indicator, OrIndicator
from tci.indicators.classical import MinmodIndicator, KXRCFIndicator
from tci.indicators.pa import PAIndicator
from tci.indicators.classical2d import KXRCFIndicator2D, MinmodIndicator2D

__all__ = [
    "Indicator",
    "OrIndicator",
    "MinmodIndicator",
    "KXRCFIndicator",
    "PAIndicator",
    "MinmodIndicator2D",
    "KXRCFIndicator2D",
]


def get_indicator(name, **kwargs):
    """Build an indicator by name; the GNN indicator needs model_path."""
    name = name.lower()
    if name == "minmod":
        return MinmodIndicator(**kwargs)
    if name == "kxrcf":
        return KXRCFIndicator(**kwargs)
    if name == "pa":
        return PAIndicator(**kwargs)
    if name == "gnn":
        from tci.indicators.learned import GNNIndicator

        return GNNIndicator(**kwargs)
    if name == "mlp":
        from tci.indicators.learned import MLPIndicator

        return MLPIndicator(**kwargs)
    if name == "gnn2d":
        from tci.indicators.learned import GNN2DIndicator

        return GNN2DIndicator(**kwargs)
    if name == "minmod2d":
        return MinmodIndicator2D(**kwargs)
    if name == "kxrcf2d":
        return KXRCFIndicator2D(**kwargs)
    if name == "mlp2d":
        from tci.indicators.learned import MLP2DIndicator

        return MLP2DIndicator(**kwargs)
    raise ValueError(f"unknown indicator {name!r}")

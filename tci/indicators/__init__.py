from tci.indicators.base import Indicator
from tci.indicators.classical import MinmodIndicator, KXRCFIndicator
from tci.indicators.pa import PAIndicator

__all__ = ["Indicator", "MinmodIndicator", "KXRCFIndicator", "PAIndicator"]


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
    raise ValueError(f"unknown indicator {name!r}")

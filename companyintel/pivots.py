"""Code-owned pivot registry adapter."""
from tools.companyintel_pivots import PivotSpec, expand_pivots, get_pivot

def registry_for(node_type: str) -> tuple[PivotSpec, ...]:
    return expand_pivots(node_type)

__all__ = ["PivotSpec", "expand_pivots", "get_pivot", "registry_for"]

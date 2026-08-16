"""Document metadata policy; never executes document payloads."""
from pathlib import Path

def describe(path: str) -> dict[str, object]:
    item = Path(path)
    return {"name": item.name, "suffix": item.suffix.lower(), "exists": item.is_file(), "size": item.stat().st_size if item.is_file() else 0}

"""Image/favicon metadata helpers."""
from urllib.parse import urlparse

def image_kind(url: str) -> str:
    path = (urlparse(url).path or "").lower()
    if "favicon" in path or path.endswith(".ico"): return "favicon"
    return "image"

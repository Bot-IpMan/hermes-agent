"""JavaScript identifier extraction without execution."""
import re

def public_identifiers(source: str) -> list[str]:
    return sorted(set(re.findall(r"(?:analytics|merchant|app|client)[_-]?(?:id|key)\s*[:=]\s*['\"]([^'\"]+)", source, re.I)))[:100]

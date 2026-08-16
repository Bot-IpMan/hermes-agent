"""Bounded HTML extraction helpers."""
from html import unescape
import re

def text_digest(html: str, limit: int = 2000) -> str:
    text = re.sub(r"<script\b[^>]*>.*?</script>|<style\b[^>]*>.*?</style>", " ", html, flags=re.I | re.S)
    return unescape(re.sub(r"<[^>]+>", " ", text)).strip()[:limit]

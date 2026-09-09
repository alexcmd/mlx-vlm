"""Text normalization."""
import re
import unicodedata

_WS = re.compile(r"\s+")

def collapse_whitespace(text: str) -> str:
    """Collapses any run of whitespace into a single space."""
    return _WS.sub(" ", text).strip()

def strip_accents(text: str) -> str:
    """Removes diacritics, keeping the base characters."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))

def normalize(text: str) -> str:
    """Full normalization: case, diacritics, whitespace."""
    return collapse_whitespace(strip_accents(text.lower()))

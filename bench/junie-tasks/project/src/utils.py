"""Helper functions."""
import re

from src.normalize import normalize

_KEEP = re.compile(r"[^\w-]|_")


def slugify(text: str) -> str:
    """Turns text into a URL slug: spaces to dashes, no special characters."""
    dashed = normalize(text).replace(" ", "-")
    return _KEEP.sub("", dashed)


def chunked(items: list, size: int) -> list[list]:
    """Splits a list into chunks of the given size."""
    return [items[i:i+size] for i in range(0, len(items), size)]

def flatten(nested: list[list]) -> list:
    """Flattens a list of lists. TODO: only one level of nesting."""
    return [x for sub in nested for x in sub]

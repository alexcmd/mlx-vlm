"""Splitting text into tokens."""
import re
from typing import Iterator

_WORD = re.compile(r"[\w']+", re.UNICODE)

def words(text: str) -> list[str]:
    """Returns the list of words."""
    return _WORD.findall(text)

def sentences(text: str) -> Iterator[str]:
    """Iterator over sentences. TODO: abbreviations like "e.g." break the split."""
    for part in re.split(r"(?<=[.!?])\s+", text):
        if part.strip():
            yield part.strip()

def ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    """All n-grams of a token list."""
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

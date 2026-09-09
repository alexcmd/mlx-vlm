"""Text statistics."""
from collections import Counter

def word_freq(words: list[str]) -> Counter:
    """Word frequencies."""
    return Counter(words)

def top_n(freq: Counter, n: int = 10) -> list[tuple[str, int]]:
    """N most frequent words. TODO: order is non-deterministic for equal counts."""
    return freq.most_common(n)

def lexical_diversity(words: list[str]) -> float:
    """Ratio of unique words to the total count."""
    return len(set(words)) / len(words) if words else 0.0

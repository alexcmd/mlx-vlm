from src.tokenize import words, ngrams

def test_words():
    assert words("one, two three") == ["one", "two", "three"]

def test_ngrams():
    assert ngrams(["a","b","c"], 2) == [("a","b"), ("b","c")]

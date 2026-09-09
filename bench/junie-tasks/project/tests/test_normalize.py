from src.normalize import collapse_whitespace, strip_accents, normalize

def test_collapse():
    assert collapse_whitespace("  a   b  ") == "a b"

def test_accents():
    assert strip_accents("café") == "cafe"

def test_normalize():
    assert normalize("  Café   NOIR ") == "cafe noir"

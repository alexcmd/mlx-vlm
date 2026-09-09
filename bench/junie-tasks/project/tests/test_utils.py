from src.utils import slugify


def test_slugify_basic():
    assert slugify("Café NOIR") == "cafe-noir"


def test_slugify_punctuation():
    assert slugify("Hello,  World!") == "hello-world"


def test_slugify_keeps_digits():
    assert slugify("room 104 b") == "room-104-b"


def test_slugify_removes_underscore():
    assert slugify("foo_bar") == "foobar"


def test_slugify_accents():
    assert slugify("Über Straße") == "uber-straße"

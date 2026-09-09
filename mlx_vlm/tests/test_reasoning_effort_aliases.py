from mlx_vlm.server.request_normalization import _reasoning_effort_enabled


def test_high_maps_to_xhigh():
    assert _reasoning_effort_enabled("high") == (True, "xhigh")
    assert _reasoning_effort_enabled(" High ") == (True, "xhigh")


def test_minimal_maps_to_low():
    assert _reasoning_effort_enabled("minimal") == (True, "low")


def test_none_stays_disabled_but_normalizes_to_low():
    assert _reasoning_effort_enabled("none") == (False, "low")


def test_known_levels_untouched():
    for level in ("low", "medium", "xhigh"):
        assert _reasoning_effort_enabled(level) == (True, level)
    assert _reasoning_effort_enabled("off") == (False, "off")
    assert _reasoning_effort_enabled(None) == (None, None)
    assert _reasoning_effort_enabled("  ") == (None, None)

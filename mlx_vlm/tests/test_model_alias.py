import sys
from types import SimpleNamespace

import pytest

import mlx_vlm.server.app  # noqa: F401  (the package attribute is the FastAPI object)

server_app = sys.modules["mlx_vlm.server.app"]


class _Registry:
    def __init__(self, loaded):
        self._loaded = loaded

    def for_kind(self, kind):
        if kind == "text_generation" and self._loaded:
            return {"cache_key": (self._loaded, None, "text_generation")}
        return {}


@pytest.fixture
def loaded_model(monkeypatch):
    monkeypatch.setattr(server_app, "_model_cache_registry", lambda: _Registry("org/served-4bit"))
    monkeypatch.setattr(server_app, "_model_is_local", lambda name: name.startswith("local/"))
    monkeypatch.delenv("MLX_VLM_MODEL_ALIAS", raising=False)
    server_app._aliased_model_names_logged.clear()
    return "org/served-4bit"


def test_unknown_name_maps_to_loaded_text_model(loaded_model):
    assert server_app._alias_model_path("vendor-cloud-model-xl") == loaded_model
    assert server_app._alias_model_path("Qwen3.8-27B-MLX-4bit", "text_generation") == loaded_model


def test_local_and_loaded_names_are_kept(loaded_model):
    assert server_app._alias_model_path(loaded_model) == loaded_model
    assert server_app._alias_model_path("local/other") == "local/other"


def test_non_text_kinds_and_opt_out_are_untouched(loaded_model, monkeypatch):
    assert server_app._alias_model_path("some/image-model", "image_generation") == "some/image-model"
    monkeypatch.setenv("MLX_VLM_MODEL_ALIAS", "0")
    assert server_app._alias_model_path("vendor-cloud-model-xl") == "vendor-cloud-model-xl"


def test_nothing_loaded_means_no_alias(monkeypatch):
    monkeypatch.setattr(server_app, "_model_cache_registry", lambda: _Registry(None))
    monkeypatch.delenv("MLX_VLM_MODEL_ALIAS", raising=False)
    assert server_app._alias_model_path("vendor-cloud-model-xl") == "vendor-cloud-model-xl"

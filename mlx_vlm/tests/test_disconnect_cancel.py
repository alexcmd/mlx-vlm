import asyncio
import sys

import pytest

import mlx_vlm.server.app  # noqa: F401  (the package attribute is the FastAPI object)
from mlx_vlm.server.generation import request_cancel_registry

server_app = sys.modules["mlx_vlm.server.app"]
DisconnectCancelMiddleware = server_app.DisconnectCancelMiddleware


def _scope(path="/v1/chat/completions"):
    return {"type": "http", "path": path, "method": "POST", "headers": []}


def _run(app_body, messages, path="/v1/chat/completions"):
    """Drive the middleware with a scripted receive channel; returns sent messages."""
    sent = []
    queue = asyncio.Queue()
    for m in messages:
        queue.put_nowait(m)

    async def receive():
        return await queue.get()

    async def send(message):
        sent.append(message)

    async def main():
        mw = DisconnectCancelMiddleware(app_body)
        await mw(_scope(path), receive, send)

    asyncio.run(main())
    return sent


def test_disconnect_after_body_cancels_registered_generation(monkeypatch):
    monkeypatch.delenv("MLX_VLM_CANCEL_ON_DISCONNECT", raising=False)
    cancelled = []

    async def app_body(scope, receive, send):
        await receive()  # the request body
        registry = request_cancel_registry.get()
        registry["cancels"].append(lambda: cancelled.append("closed"))
        # simulate a long generation: the client disconnects meanwhile
        for _ in range(40):
            if cancelled:
                break
            await asyncio.sleep(0.02)
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"{}", "more_body": False})

    sent = _run(app_body, [{"type": "http.request", "body": b"{}", "more_body": False}, {"type": "http.disconnect"}])
    assert cancelled == ["closed"]
    assert sent[-1]["type"] == "http.response.body"
    assert request_cancel_registry.get() is None  # context restored


def test_late_registration_after_disconnect_is_closed_immediately(monkeypatch):
    monkeypatch.delenv("MLX_VLM_CANCEL_ON_DISCONNECT", raising=False)
    from queue import Queue

    from mlx_vlm.server.generation import _TokenIterator

    cancelled = []

    async def app_body(scope, receive, send):
        await receive()
        await asyncio.sleep(0.15)  # disconnect arrives while "prefilling"
        it = _TokenIterator(Queue(), uid=3, cancel_fn=cancelled.append, queue_timeout=1)
        assert it._closed
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"", "more_body": False})

    _run(app_body, [{"type": "http.request", "body": b"{}", "more_body": False}, {"type": "http.disconnect"}])
    assert cancelled == [3]


def test_completed_request_is_not_cancelled_and_other_paths_pass_through(monkeypatch):
    monkeypatch.delenv("MLX_VLM_CANCEL_ON_DISCONNECT", raising=False)
    cancelled = []

    async def app_body(scope, receive, send):
        await receive()
        registry = request_cancel_registry.get()
        if registry is not None:
            registry["cancels"].append(lambda: cancelled.append(scope["path"]))
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok", "more_body": False})

    _run(app_body, [{"type": "http.request", "body": b"{}", "more_body": False}, {"type": "http.disconnect"}])
    assert cancelled == []
    seen = {}

    async def other(scope, receive, send):
        seen["registry"] = request_cancel_registry.get()
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"", "more_body": False})

    _run(other, [{"type": "http.request", "body": b"", "more_body": False}], path="/v1/models")
    assert seen["registry"] is None


def test_can_be_disabled(monkeypatch):
    monkeypatch.setenv("MLX_VLM_CANCEL_ON_DISCONNECT", "0")
    seen = {}

    async def app_body(scope, receive, send):
        seen["registry"] = request_cancel_registry.get()
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"", "more_body": False})

    _run(app_body, [{"type": "http.request", "body": b"", "more_body": False}])
    assert seen["registry"] is None

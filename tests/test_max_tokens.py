"""Regression tests ensuring caller max_tokens budgets are never inflated.

Includes stdlib-only AST-loaded unit tests (so minimal CI runners execute them
without external dependencies) plus end-to-end route handler tests whenever
FastAPI and httpx are installed.
"""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import re
import sys
from typing import Any
import unittest
from unittest.mock import AsyncMock, patch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "gateway.py"


def load_max_tokens_helpers() -> dict[str, Any]:
    """Load _resolve_max_tokens and the messages request translator via AST."""
    tree = ast.parse(SOURCE.read_text(), filename=str(SOURCE))
    selected = []
    translator_name = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in {
            "_resolve_max_tokens",
            "_content_blocks_to_text",
        }:
            selected.append(node)
        elif (
            isinstance(node, ast.FunctionDef)
            and node.name.endswith("_request")
            and "stop_sequences" in ast.dump(node)
        ):
            selected.append(node)
            translator_name = node.name
    namespace: dict[str, Any] = {
        "Any": Any,
        "DEFAULT_MAX_TOKENS": 16384,
        "json": json,
        "re": re,
    }
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(SOURCE), "exec"), namespace)
    assert translator_name is not None, "messages request translator not found"
    namespace["translate_messages_request"] = namespace[translator_name]
    return namespace


class ResolveMaxTokensTests(unittest.TestCase):
    def setUp(self):
        self.ns = load_max_tokens_helpers()
        self.resolve = self.ns["_resolve_max_tokens"]
        self.translate = self.ns["translate_messages_request"]

    def test_explicit_small_max_tokens_is_preserved_verbatim(self):
        for budget in (1, 64, 128, 256, 512, 1024, 4096):
            with self.subTest(max_tokens=budget):
                self.assertEqual(budget, self.resolve(budget))

    def test_omitted_or_none_max_tokens_gets_default(self):
        self.assertEqual(16384, self.resolve(None))
        self.assertEqual(16384, self.resolve({}.get("max_tokens")))

    def test_larger_max_tokens_is_preserved_unless_above_ceiling(self):
        self.assertEqual(16384, self.resolve(16384))
        self.assertEqual(32768, self.resolve(32768))
        self.assertEqual(65536, self.resolve(65536))
        self.assertEqual(32768, self.resolve(65536, ceiling=32768))
        # Ceiling clamps down only; never bumps up a smaller explicit budget.
        self.assertEqual(512, self.resolve(512, ceiling=32768))

    def test_non_positive_or_invalid_values_fall_back_to_default(self):
        for raw in (0, -1, -256, False, True, "", "not-a-number"):
            with self.subTest(raw=raw):
                self.assertEqual(16384, self.resolve(raw))
        self.assertEqual(256, self.resolve("256"))

    def test_messages_translator_respects_caller_max_tokens(self):
        base = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
        }
        for budget in (128, 512, 1024, 4096, 32768):
            with self.subTest(max_tokens=budget):
                out = self.translate({**base, "max_tokens": budget})
                self.assertEqual(budget, out["max_tokens"])

        self.assertEqual(16384, self.translate(dict(base))["max_tokens"])
        self.assertEqual(16384, self.translate({**base, "max_tokens": None})["max_tokens"])


try:
    import fastapi  # noqa: F401
    import httpx  # noqa: F401

    _HAS_WEB_DEPS = True
except Exception:
    _HAS_WEB_DEPS = False


if _HAS_WEB_DEPS:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    os.environ.setdefault("GATEWAY_MODE", "walter")
    os.environ.setdefault("TRACING_ENABLED", "false")
    os.environ.setdefault("GATEWAY_CONFIG", "/tmp/gordon-gateway-test-no-config.yaml")
    os.environ.setdefault("LOG_FILE", "/tmp/gordon-gateway-test.log")
    os.environ.setdefault("BLOCKOPS_LOG_FILE", "/tmp/gordon-gateway-blockops-test.log")

    from fastapi import Request
    import gateway


def _make_json_request(path: str, payload: dict[str, Any]) -> "Request":
    body = json.dumps(payload).encode("utf-8")
    delivered = False

    async def receive():
        nonlocal delivered
        if delivered:
            return {"type": "http.disconnect"}
        delivered = True
        return {"type": "http.request", "body": body, "more_body": False}

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode("ascii"),
        "query_string": b"",
        "root_path": "",
        "headers": [(b"content-type", b"application/json")],
        "client": ("127.0.0.1", 12345),
        "server": ("127.0.0.1", 8091),
    }
    return Request(scope, receive)


class _DummyResponse:
    def __init__(self, payload: dict[str, Any], status_code: int = 200):
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload)

    def json(self) -> dict[str, Any]:
        return self._payload


@unittest.skipUnless(_HAS_WEB_DEPS, "fastapi and httpx required for handler tests")
class RouteMaxTokensTests(unittest.IsolatedAsyncioTestCase):
    async def _run_chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
        captured: dict[str, Any] = {}
        backend = gateway.Backend(
            name="unit-backend",
            base_url="http://127.0.0.1:9999/v1",
            upstream_model="unit-upstream",
            api_key="not-needed",
        )

        class FakeAsyncClient:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def post(self, url, json=None, headers=None):
                captured["url"] = url
                captured["json"] = dict(json or {})
                return _DummyResponse({
                    "id": "chatcmpl-test",
                    "model": "unit-upstream",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
                })

        req = _make_json_request("/v1/chat/completions", payload)
        with patch.object(gateway, "MODE", "alfred"), \
             patch.object(gateway, "MASTER_KEY", "test-master-key"), \
             patch.object(gateway, "BACKENDS", {"unit-backend": backend}), \
             patch.object(gateway, "FALLBACKS", {}), \
             patch.object(gateway, "TRACING_ENABLED", False), \
             patch.object(gateway, "verify_swap_in_loaded", AsyncMock(return_value=True)), \
             patch.object(gateway.httpx, "AsyncClient", FakeAsyncClient):
            resp = await gateway.chat_completions(req, authorization="Bearer test-master-key")
        self.assertEqual(200, resp.status_code)
        return captured["json"]

    async def _run_messages(self, payload: dict[str, Any]) -> dict[str, Any]:
        captured: dict[str, Any] = {}
        backend = gateway.Backend(
            name="unit-backend",
            base_url="http://127.0.0.1:9999/v1",
            upstream_model="unit-upstream",
            api_key="not-needed",
        )

        class FakeAsyncClient:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def post(self, url, json=None, headers=None):
                captured["url"] = url
                captured["json"] = dict(json or {})
                return _DummyResponse({
                    "id": "chatcmpl-test",
                    "model": "unit-upstream",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
                })

        req = _make_json_request("/v1/messages", payload)
        with patch.object(gateway, "MODE", "alfred"), \
             patch.object(gateway, "MASTER_KEY", "test-master-key"), \
             patch.object(gateway, "BACKENDS", {"unit-backend": backend}), \
             patch.object(gateway, "FALLBACKS", {}), \
             patch.object(gateway, "TRACING_ENABLED", False), \
             patch.object(gateway, "verify_swap_in_loaded", AsyncMock(return_value=True)), \
             patch.object(gateway.httpx, "AsyncClient", FakeAsyncClient):
            resp = await gateway.messages(req, authorization="Bearer test-master-key", x_api_key=None)
        self.assertEqual(200, resp.status_code)
        return captured["json"]

    async def _run_walter_tools(self, extra_fields: dict[str, Any]) -> dict[str, Any]:
        captured: dict[str, Any] = {}

        async def fake_fetch(out_body, headers, path, alias=None):
            captured["json"] = json.loads(out_body.decode("utf-8"))
            resp = {
                "choices": [{"message": {"role": "assistant", "content": "done"}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
            }
            return json.dumps(resp).encode("utf-8"), 200

        messages = [{"role": "user", "content": "hello"}]
        tools = [{"type": "function", "function": {"name": "ping", "description": "ping", "parameters": {}}}]
        data: dict[str, Any] = {"model": "qwen35", "messages": list(messages), "tools": tools, **extra_fields}
        req = _make_json_request("/v1/chat/completions", data)
        session_state = gateway.WalterSessionState()
        with patch.object(gateway, "fetch_backend_blocking", fake_fetch), \
             patch.object(gateway, "bolog"), \
             patch.object(gateway, "_walter_trace"):
            stream_resp = await gateway._walter_tools_path(
                req,
                data,
                {},
                list(messages),
                tools,
                "http://127.0.0.1:8082",
                False,
                b"{}",
                "tester",
                "primary",
                0.0,
                session_state,
            )
            async for _ in stream_resp.body_iterator:
                pass
        return captured["json"]

    async def test_chat_completions_preserves_small_and_large_and_defaults_missing(self):
        base = {"model": "unit-backend", "messages": [{"role": "user", "content": "hello"}]}
        for budget in (128, 512, 1024, 4096, 32768):
            with self.subTest(max_tokens=budget):
                sent = await self._run_chat_completions({**base, "max_tokens": budget})
                self.assertEqual(budget, sent["max_tokens"])

        sent_omitted = await self._run_chat_completions(dict(base))
        self.assertEqual(16384, sent_omitted["max_tokens"])

        sent_none = await self._run_chat_completions({**base, "max_tokens": None})
        self.assertEqual(16384, sent_none["max_tokens"])

    async def test_messages_route_preserves_small_and_large_and_defaults_missing(self):
        base = {"model": "unit-backend", "messages": [{"role": "user", "content": "hello"}]}
        for budget in (128, 1024, 32768):
            with self.subTest(max_tokens=budget):
                sent = await self._run_messages({**base, "max_tokens": budget})
                self.assertEqual(budget, sent["max_tokens"])

        sent_omitted = await self._run_messages(dict(base))
        self.assertEqual(16384, sent_omitted["max_tokens"])

        sent_none = await self._run_messages({**base, "max_tokens": None})
        self.assertEqual(16384, sent_none["max_tokens"])

    async def test_walter_tools_path_preserves_small_and_large_and_defaults_missing(self):
        for budget in (128, 1024, 32768):
            with self.subTest(max_tokens=budget):
                sent = await self._run_walter_tools({"max_tokens": budget})
                self.assertEqual(budget, sent["max_tokens"])

        sent_omitted = await self._run_walter_tools({})
        self.assertEqual(16384, sent_omitted["max_tokens"])

        sent_none = await self._run_walter_tools({"max_tokens": None})
        self.assertEqual(16384, sent_none["max_tokens"])


if __name__ == "__main__":
    unittest.main()

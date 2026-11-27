"""Comprehensive offline regression tests for GW-01 through GW-07.

Covers:
  - GW-01: Incremental LF/CRLF/CR SSE framing, arbitrary chunk fragmentation,
    content/reasoning/tool-call preservation, and premature/empty EOF failure.
  - GW-02: Terminal HTTP 4xx default vs explicit opt-in, 5xx/transport failover,
    and blocked cross-boundary external fallback unless explicitly authorized.
  - GW-03: Non-JSON HTTP 200, malformed JSON shape, non-SSE buffered garbage,
    and httpx.ConnectTimeout failover + typed 502 error when exhausted.
  - GW-04: Normalized completion representation parity between buffered JSON
    translation and fake-SSE stream adaptation (preserving reasoning_content).
  - GW-05: Anthropic tool_choice 'none', base64/URL image preservation, typed
    400 rejection of unsupported blocks/controls, and Walter tool_choice 'none'.
  - GW-06: Metadata-only telemetry default (TRACE_CONTENT=False), authenticated
    TLS enforcement for non-loopback OTLP endpoints, and secret-file alignment.
  - GW-07: Generic public defaults (no embedded deployment identifiers) and
    declared dependency ranges.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import sys
from typing import Any, AsyncIterator
import unittest
from unittest.mock import AsyncMock, patch


ROOT = Path(__file__).resolve().parents[1]

try:
    import fastapi  # noqa: F401
    import httpx
    from fastapi import Request

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
    import gateway


def _make_json_request(path: str, payload: dict[str, Any], extra_headers: list[tuple[bytes, bytes]] | None = None) -> "Request":
    body = json.dumps(payload).encode("utf-8")
    delivered = False

    async def receive():
        nonlocal delivered
        if delivered:
            return {"type": "http.disconnect"}
        delivered = True
        return {"type": "http.request", "body": body, "more_body": False}

    headers = [(b"content-type", b"application/json")]
    if extra_headers:
        headers.extend(extra_headers)

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
        "headers": headers,
        "client": ("127.0.0.1", 12345),
        "server": ("127.0.0.1", 8091),
    }
    return Request(scope, receive)


async def _chunked_iter(payload: bytes, chunk_size: int) -> AsyncIterator[bytes]:
    for i in range(0, len(payload), chunk_size):
        yield payload[i:i + chunk_size]


async def _collect_stream(async_iter: AsyncIterator[bytes]) -> bytes:
    parts: list[bytes] = []
    async for chunk in async_iter:
        parts.append(chunk)
    return b"".join(parts)


def _parse_output_sse(raw: bytes) -> list[tuple[str, str]]:
    events, tail = gateway.parse_sse_blocks(raw, final=True)
    assert not gateway._has_unparsed_sse_tail(tail), f"unexpected unparsed tail: {tail!r}"
    return events


def _reconstruct_anthropic_from_sse(raw: bytes) -> dict[str, Any]:
    """Reconstruct an Anthropic Message object from Anthropic SSE events."""
    events = _parse_output_sse(raw)
    msg: dict[str, Any] = {"content": [], "stop_reason": None, "usage": {"input_tokens": 0, "output_tokens": 0}}
    blocks: dict[int, dict[str, Any]] = {}
    tool_json: dict[int, str] = {}
    saw_stop = False
    for etype, data_str in events:
        evt = json.loads(data_str)
        if etype == "message_start":
            msg["model"] = evt["message"]["model"]
        elif etype == "content_block_start":
            idx = evt["index"]
            cb = dict(evt["content_block"])
            blocks[idx] = cb
            if cb.get("type") == "tool_use":
                tool_json[idx] = ""
        elif etype == "content_block_delta":
            idx = evt["index"]
            delta = evt["delta"]
            if delta["type"] == "thinking_delta":
                blocks[idx]["thinking"] += delta["thinking"]
            elif delta["type"] == "text_delta":
                blocks[idx]["text"] += delta["text"]
            elif delta["type"] == "input_json_delta":
                tool_json[idx] += delta["partial_json"]
        elif etype == "content_block_stop":
            idx = evt["index"]
            if blocks[idx].get("type") == "tool_use":
                blocks[idx]["input"] = json.loads(tool_json[idx] or "{}")
        elif etype == "message_delta":
            msg["stop_reason"] = evt["delta"]["stop_reason"]
            msg["usage"] = evt["usage"]
        elif etype == "message_stop":
            saw_stop = True
    msg["content"] = [blocks[i] for i in sorted(blocks)]
    msg["_saw_message_stop"] = saw_stop
    return msg


class _FakeStreamResponse:
    def __init__(
        self,
        *,
        status_code: int = 200,
        headers: dict[str, str] | None = None,
        chunks: list[bytes] | None = None,
        body: bytes = b"",
        midstream_exc: Exception | None = None,
    ):
        self.status_code = status_code
        self.headers = headers or {"content-type": "text/event-stream"}
        self._chunks = chunks if chunks is not None else ([body] if body else [])
        self._body = body or b"".join(self._chunks)
        self._midstream_exc = midstream_exc
        self.closed = False

    async def aread(self) -> bytes:
        return self._body

    async def aiter_raw(self) -> AsyncIterator[bytes]:
        for c in self._chunks:
            yield c
        if self._midstream_exc is not None:
            raise self._midstream_exc

    async def aclose(self) -> None:
        self.closed = True


class _FakePostResponse:
    def __init__(
        self,
        *,
        status_code: int = 200,
        body: str = "",
        headers: dict[str, str] | None = None,
    ):
        self.status_code = status_code
        self.text = body
        self.headers = headers or {"content-type": "application/json"}

    def json(self) -> Any:
        return json.loads(self.text)


def _ok_completion_json(content: str = "ok", model: str = "upstream-m", reasoning: str | None = None) -> str:
    msg: dict[str, Any] = {"role": "assistant", "content": content}
    if reasoning is not None:
        msg["reasoning_content"] = reasoning
    return json.dumps({
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1700000000,
        "model": model,
        "choices": [{"index": 0, "message": msg, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    })


def _make_fake_httpx_client(outcomes_by_url: dict[str, Any], calls: list[dict[str, Any]]):
    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def aclose(self):
            return None

        def build_request(self, method: str, url: str, json: Any = None, headers: Any = None):
            return {"method": method, "url": url, "json": json, "headers": headers}

        async def send(self, req: dict[str, Any], stream: bool = False):
            calls.append({"url": req["url"], "json": req["json"], "stream": stream})
            outcome = outcomes_by_url[req["url"]]
            if callable(outcome):
                outcome = outcome(req)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        async def post(self, url: str, json: Any = None, headers: Any = None):
            calls.append({"url": url, "json": json, "stream": False})
            outcome = outcomes_by_url[url]
            if callable(outcome):
                outcome = outcome({"url": url, "json": json, "headers": headers})
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

    return FakeAsyncClient


# ===========================================================================
# GW-01: SSE framing (LF & CRLF fragmentation) + premature/empty EOF failure
# ===========================================================================
@unittest.skipUnless(_HAS_WEB_DEPS, "fastapi and httpx required")
class SSEFramingAndCompletionTests(unittest.IsolatedAsyncioTestCase):
    def _build_full_sse(self, sep: bytes) -> bytes:
        chunks = [
            {
                "id": "chatcmpl-sse",
                "object": "chat.completion.chunk",
                "model": "upstream-model",
                "choices": [{"index": 0, "delta": {"role": "assistant", "reasoning_content": "step 1"}, "finish_reason": None}],
            },
            {
                "id": "chatcmpl-sse",
                "object": "chat.completion.chunk",
                "model": "upstream-model",
                "choices": [{"index": 0, "delta": {"content": "hello "}, "finish_reason": None}],
            },
            {
                "id": "chatcmpl-sse",
                "object": "chat.completion.chunk",
                "model": "upstream-model",
                "choices": [{"index": 0, "delta": {"content": "world"}, "finish_reason": None}],
            },
            {
                "id": "chatcmpl-sse",
                "object": "chat.completion.chunk",
                "model": "upstream-model",
                "choices": [{
                    "index": 0,
                    "delta": {
                        "tool_calls": [{
                            "index": 0,
                            "id": "call_abc",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": '{"sym":'},
                        }]
                    },
                    "finish_reason": None,
                }],
            },
            {
                "id": "chatcmpl-sse",
                "object": "chat.completion.chunk",
                "model": "upstream-model",
                "choices": [{
                    "index": 0,
                    "delta": {
                        "tool_calls": [{
                            "index": 0,
                            "function": {"arguments": '"AAPL"}'},
                        }]
                    },
                    "finish_reason": "tool_calls",
                }],
                "usage": {"prompt_tokens": 12, "completion_tokens": 8},
            },
        ]
        parts = [b": keepalive" + sep + sep]
        for ch in chunks:
            parts.append(b"data: " + json.dumps(ch).encode("utf-8") + sep + sep)
        parts.append(b"data: [DONE]" + sep + sep)
        return b"".join(parts)

    async def test_fragmented_lf_and_crlf_preserve_reasoning_content_and_tools_anthropic(self):
        for label, sep in (("lf", b"\n"), ("crlf", b"\r\n")):
            raw_sse = self._build_full_sse(sep)
            for chunk_size in (1, 3, 7, 29, len(raw_sse)):
                with self.subTest(line_ending=label, chunk_size=chunk_size):
                    out = await _collect_stream(
                        gateway.openai_stream_to_anthropic(_chunked_iter(raw_sse, chunk_size), "requested-model")
                    )
                    reconstructed = _reconstruct_anthropic_from_sse(out)
                    self.assertTrue(reconstructed["_saw_message_stop"])
                    self.assertEqual("requested-model", reconstructed["model"])
                    self.assertEqual("tool_use", reconstructed["stop_reason"])
                    self.assertEqual({"input_tokens": 12, "output_tokens": 8}, reconstructed["usage"])
                    self.assertEqual(
                        [
                            {"type": "thinking", "thinking": "step 1"},
                            {"type": "text", "text": "hello world"},
                            {"type": "tool_use", "id": "call_abc", "name": "lookup", "input": {"sym": "AAPL"}},
                        ],
                        reconstructed["content"],
                    )

    async def test_fragmented_lf_and_crlf_preserve_payload_and_rewrite_model_openai(self):
        for label, sep in (("lf", b"\n"), ("crlf", b"\r\n")):
            raw_sse = self._build_full_sse(sep)
            for chunk_size in (1, 5, 17):
                with self.subTest(line_ending=label, chunk_size=chunk_size):
                    out = await _collect_stream(
                        gateway.rewrite_stream_model(_chunked_iter(raw_sse, chunk_size), "requested-model")
                    )
                    events = _parse_output_sse(out)
                    self.assertEqual("[DONE]", events[-1][1])
                    self.assertNotIn("error", [et for et, _ in events])
                    objs = [json.loads(d) for _, d in events[:-1]]
                    self.assertTrue(all(o["model"] == "requested-model" for o in objs))
                    reasoning = "".join(o["choices"][0]["delta"].get("reasoning_content", "") for o in objs)
                    content = "".join(o["choices"][0]["delta"].get("content", "") for o in objs)
                    args = "".join(
                        tc.get("function", {}).get("arguments", "")
                        for o in objs
                        for tc in (o["choices"][0]["delta"].get("tool_calls") or [])
                    )
                    self.assertEqual("step 1", reasoning)
                    self.assertEqual("hello world", content)
                    self.assertEqual({"sym": "AAPL"}, json.loads(args))

    async def test_empty_stream_never_reports_success(self):
        out_anthropic = await _collect_stream(gateway.openai_stream_to_anthropic(_chunked_iter(b"", 1), "m"))
        events_a = _parse_output_sse(out_anthropic)
        self.assertEqual(["error"], [et for et, _ in events_a])
        self.assertNotIn(b"message_stop", out_anthropic)
        self.assertNotIn(b"end_turn", out_anthropic)

        out_openai = await _collect_stream(gateway.rewrite_stream_model(_chunked_iter(b"", 1), "m"))
        events_o = _parse_output_sse(out_openai)
        self.assertEqual(["error"], [et for et, _ in events_o])
        self.assertNotIn(b"[DONE]", out_openai)

    async def test_done_only_stream_never_reports_success(self):
        raw = b"data: [DONE]\r\n\r\n"
        out_a = await _collect_stream(gateway.openai_stream_to_anthropic(_chunked_iter(raw, 2), "m"))
        self.assertIn("error", [et for et, _ in _parse_output_sse(out_a)])
        self.assertNotIn(b"message_stop", out_a)

        out_o = await _collect_stream(gateway.rewrite_stream_model(_chunked_iter(raw, 2), "m"))
        self.assertIn("error", [et for et, _ in _parse_output_sse(out_o)])
        self.assertNotIn(b"[DONE]", out_o)

    async def test_premature_eof_without_finish_or_done_emits_error_not_success(self):
        partial = (
            b'data: {"id":"c1","model":"up","choices":[{"index":0,"delta":{"content":"partial"},"finish_reason":null}]}\r\n\r\n'
        )
        out_a = await _collect_stream(gateway.openai_stream_to_anthropic(_chunked_iter(partial, 3), "m"))
        etypes_a = [et for et, _ in _parse_output_sse(out_a)]
        self.assertEqual("error", etypes_a[-1])
        self.assertNotIn("message_stop", etypes_a)
        self.assertNotIn("message_delta", etypes_a)

        out_o = await _collect_stream(gateway.rewrite_stream_model(_chunked_iter(partial, 3), "m"))
        etypes_o = [et for et, _ in _parse_output_sse(out_o)]
        self.assertEqual("error", etypes_o[-1])
        self.assertNotIn(b"[DONE]", out_o)

    async def test_unterminated_trailing_bytes_emits_error(self):
        truncated = (
            b'data: {"id":"c1","model":"up","choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":"stop"}]}\n\n'
            b'data: {"truncated_json'
        )
        out_o = await _collect_stream(gateway.rewrite_stream_model(_chunked_iter(truncated, 7), "m"))
        self.assertEqual("error", _parse_output_sse(out_o)[-1][0])

        out_a = await _collect_stream(gateway.openai_stream_to_anthropic(_chunked_iter(truncated, 7), "m"))
        etypes_a = [et for et, _ in _parse_output_sse(out_a)]
        self.assertEqual("error", etypes_a[-1])
        self.assertNotIn("message_stop", etypes_a)

    async def test_midstream_transport_error_emits_error_not_success(self):
        async def broken_iter() -> AsyncIterator[bytes]:
            yield b'data: {"id":"c1","model":"up","choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":null}]}\r\n\r\n'
            raise httpx.ReadError("connection reset mid-stream")

        out_a = await _collect_stream(gateway.openai_stream_to_anthropic(broken_iter(), "m"))
        etypes_a = [et for et, _ in _parse_output_sse(out_a)]
        self.assertEqual("error", etypes_a[-1])
        self.assertNotIn("message_stop", etypes_a)

        out_o = await _collect_stream(gateway.rewrite_stream_model(broken_iter(), "m"))
        etypes_o = [et for et, _ in _parse_output_sse(out_o)]
        self.assertEqual("error", etypes_o[-1])
        self.assertNotIn(b"[DONE]", out_o)

    async def test_empty_upstream_stream_fails_over_before_http_200_commit(self):
        primary = gateway.Backend("local-primary", "http://127.0.0.1:9001/v1", "p-model", "k")
        backup = gateway.Backend("local-backup", "http://127.0.0.1:9002/v1", "b-model", "k")
        good_crlf = (
            b'data: {"id":"c2","model":"b-model","choices":[{"index":0,"delta":{"content":"recovered"},"finish_reason":"stop"}]}\r\n\r\n'
            b"data: [DONE]\r\n\r\n"
        )
        calls: list[dict[str, Any]] = []
        outcomes = {
            "http://127.0.0.1:9001/v1/chat/completions": _FakeStreamResponse(status_code=200, chunks=[]),
            "http://127.0.0.1:9002/v1/chat/completions": _FakeStreamResponse(status_code=200, chunks=[good_crlf[:15], good_crlf[15:]]),
        }
        req = _make_json_request(
            "/v1/messages",
            {"model": "local-primary", "stream": True, "messages": [{"role": "user", "content": "hi"}]},
        )
        with patch.object(gateway, "MODE", "alfred"), \
             patch.object(gateway, "MASTER_KEY", "secret"), \
             patch.object(gateway, "BACKENDS", {"local-primary": primary, "local-backup": backup}), \
             patch.object(gateway, "FALLBACKS", {"local-primary": ["local-backup"]}), \
             patch.object(gateway, "TRACING_ENABLED", False), \
             patch.object(gateway, "verify_swap_in_loaded", AsyncMock(return_value=True)), \
             patch.object(gateway.httpx, "AsyncClient", _make_fake_httpx_client(outcomes, calls)):
            resp = await gateway.messages(req, authorization="Bearer secret", x_api_key=None)
            self.assertEqual(200, resp.status_code)
            body = await _collect_stream(resp.body_iterator)
        self.assertEqual(2, len(calls))
        reconstructed = _reconstruct_anthropic_from_sse(body)
        self.assertTrue(reconstructed["_saw_message_stop"])
        self.assertEqual([{"type": "text", "text": "recovered"}], reconstructed["content"])


# ===========================================================================
# GW-02 & GW-03: Retry policy, 4xx terminal default, data-boundary, ConnectTimeout & malformed 200
# ===========================================================================
@unittest.skipUnless(_HAS_WEB_DEPS, "fastapi and httpx required")
class RetryAndFailoverPolicyTests(unittest.IsolatedAsyncioTestCase):
    async def test_http_400_is_terminal_by_default_and_never_reaches_second_backend(self):
        primary = gateway.Backend("local-primary", "http://127.0.0.1:9001/v1", "p-model", "k")
        backup = gateway.Backend("local-backup", "http://127.0.0.1:9002/v1", "b-model", "k")
        for endpoint, stream in (
            ("/v1/chat/completions", False),
            ("/v1/chat/completions", True),
            ("/v1/messages", False),
            ("/v1/messages", True),
        ):
            with self.subTest(endpoint=endpoint, stream=stream):
                calls: list[dict[str, Any]] = []
                err_400 = (
                    _FakeStreamResponse(status_code=400, body=b'{"error":{"message":"bad request"}}')
                    if stream
                    else _FakePostResponse(status_code=400, body='{"error":{"message":"bad request"}}')
                )
                ok_resp = (
                    _FakeStreamResponse(
                        status_code=200,
                        body=b'data: {"id":"c","model":"b","choices":[{"index":0,"delta":{"content":"nope"},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n',
                    )
                    if stream
                    else _FakePostResponse(status_code=200, body=_ok_completion_json("nope"))
                )
                outcomes = {
                    "http://127.0.0.1:9001/v1/chat/completions": err_400,
                    "http://127.0.0.1:9002/v1/chat/completions": ok_resp,
                }
                req = _make_json_request(
                    endpoint,
                    {"model": "local-primary", "stream": stream, "messages": [{"role": "user", "content": "SENSITIVE_SYNTHETIC_PROMPT"}]},
                )
                with patch.object(gateway, "MODE", "alfred"), \
                     patch.object(gateway, "MASTER_KEY", "secret"), \
                     patch.object(gateway, "BACKENDS", {"local-primary": primary, "local-backup": backup}), \
                     patch.object(gateway, "FALLBACKS", {"local-primary": ["local-backup"]}), \
                     patch.object(gateway, "RETRYABLE_4XX_STATUSES", frozenset()), \
                     patch.object(gateway, "TRACING_ENABLED", False), \
                     patch.object(gateway, "verify_swap_in_loaded", AsyncMock(return_value=True)), \
                     patch.object(gateway.httpx, "AsyncClient", _make_fake_httpx_client(outcomes, calls)):
                    if endpoint == "/v1/chat/completions":
                        resp = await gateway.chat_completions(req, authorization="Bearer secret")
                    else:
                        resp = await gateway.messages(req, authorization="Bearer secret", x_api_key=None)
                self.assertEqual(400, resp.status_code)
                self.assertEqual(1, len(calls), "400 must never retry onto backup backend by default")
                self.assertEqual("http://127.0.0.1:9001/v1/chat/completions", calls[0]["url"])

    async def test_http_400_retries_only_when_explicitly_opted_in(self):
        primary = gateway.Backend("local-primary", "http://127.0.0.1:9001/v1", "p-model", "k")
        backup = gateway.Backend("local-backup", "http://127.0.0.1:9002/v1", "b-model", "k")
        calls: list[dict[str, Any]] = []
        outcomes = {
            "http://127.0.0.1:9001/v1/chat/completions": _FakePostResponse(status_code=400, body='{"error":"retry me"}'),
            "http://127.0.0.1:9002/v1/chat/completions": _FakePostResponse(status_code=200, body=_ok_completion_json("backup-ok")),
        }
        req = _make_json_request(
            "/v1/chat/completions",
            {"model": "local-primary", "messages": [{"role": "user", "content": "hi"}]},
        )
        with patch.object(gateway, "MODE", "alfred"), \
             patch.object(gateway, "MASTER_KEY", "secret"), \
             patch.object(gateway, "BACKENDS", {"local-primary": primary, "local-backup": backup}), \
             patch.object(gateway, "FALLBACKS", {"local-primary": ["local-backup"]}), \
             patch.object(gateway, "RETRYABLE_4XX_STATUSES", frozenset({400})), \
             patch.object(gateway, "TRACING_ENABLED", False), \
             patch.object(gateway, "verify_swap_in_loaded", AsyncMock(return_value=True)), \
             patch.object(gateway.httpx, "AsyncClient", _make_fake_httpx_client(outcomes, calls)):
            resp = await gateway.chat_completions(req, authorization="Bearer secret")
        self.assertEqual(200, resp.status_code)
        self.assertEqual(2, len(calls))
        self.assertEqual("backup-ok", json.loads(resp.body)["choices"][0]["message"]["content"])

    async def test_local_to_external_fallback_is_blocked_without_explicit_authorization(self):
        local = gateway.Backend("local-primary", "http://127.0.0.1:9001/v1", "p-model", "k")
        cloud = gateway.Backend("cloud-backup", "https://openrouter.ai/api/v1", "c-model", "k")
        self.assertEqual("local", local.trust_boundary)
        self.assertEqual("external", cloud.trust_boundary)

        calls: list[dict[str, Any]] = []
        outcomes = {
            "http://127.0.0.1:9001/v1/chat/completions": _FakePostResponse(status_code=503, body="down"),
            "https://openrouter.ai/api/v1/chat/completions": _FakePostResponse(status_code=200, body=_ok_completion_json("leaked")),
        }
        req = _make_json_request(
            "/v1/chat/completions",
            {"model": "local-primary", "messages": [{"role": "user", "content": "PRIVATE_LOCAL_PROMPT"}]},
        )
        with patch.object(gateway, "MODE", "alfred"), \
             patch.object(gateway, "MASTER_KEY", "secret"), \
             patch.object(gateway, "BACKENDS", {"local-primary": local, "cloud-backup": cloud}), \
             patch.object(gateway, "FALLBACKS", {"local-primary": ["cloud-backup"]}), \
             patch.object(gateway, "ALLOW_EXTERNAL_FALLBACK", False), \
             patch.object(gateway, "EXTERNAL_FALLBACK_ROUTES", frozenset()), \
             patch.object(gateway, "TRACING_ENABLED", False), \
             patch.object(gateway, "verify_swap_in_loaded", AsyncMock(return_value=True)), \
             patch.object(gateway.httpx, "AsyncClient", _make_fake_httpx_client(outcomes, calls)):
            resp = await gateway.chat_completions(req, authorization="Bearer secret")
        self.assertEqual(502, resp.status_code)
        self.assertEqual(1, len(calls), "Prompt must never cross into external cloud backend without explicit opt-in")
        self.assertEqual("all_backends_failed", json.loads(resp.body)["error"]["code"])

        # Now authorize the route explicitly: 503 should fail over to cloud, while 400 still stops locally!
        calls.clear()
        req2 = _make_json_request(
            "/v1/chat/completions",
            {"model": "local-primary", "messages": [{"role": "user", "content": "AUTHORIZED_CLOUD_PROMPT"}]},
        )
        with patch.object(gateway, "MODE", "alfred"), \
             patch.object(gateway, "MASTER_KEY", "secret"), \
             patch.object(gateway, "BACKENDS", {"local-primary": local, "cloud-backup": cloud}), \
             patch.object(gateway, "FALLBACKS", {"local-primary": ["cloud-backup"]}), \
             patch.object(gateway, "ALLOW_EXTERNAL_FALLBACK", False), \
             patch.object(gateway, "EXTERNAL_FALLBACK_ROUTES", frozenset({"local-primary"})), \
             patch.object(gateway, "TRACING_ENABLED", False), \
             patch.object(gateway, "verify_swap_in_loaded", AsyncMock(return_value=True)), \
             patch.object(gateway.httpx, "AsyncClient", _make_fake_httpx_client(outcomes, calls)):
            resp2 = await gateway.chat_completions(req2, authorization="Bearer secret")
        self.assertEqual(200, resp2.status_code)
        self.assertEqual(2, len(calls))

    async def test_connect_timeout_and_malformed_200_trigger_failover(self):
        primary = gateway.Backend("local-primary", "http://127.0.0.1:9001/v1", "p-model", "k")
        backup = gateway.Backend("local-backup", "http://127.0.0.1:9002/v1", "b-model", "k")

        cases = [
            ("connect_timeout", httpx.ConnectTimeout("timed out connecting")),
            ("non_json_200", _FakePostResponse(status_code=200, body="<html>502 Bad Gateway</html>", headers={"content-type": "text/html"})),
            ("invalid_json_200", _FakePostResponse(status_code=200, body="not-json-at-all")),
            ("missing_choices_200", _FakePostResponse(status_code=200, body='{"id":"x"}')),
            ("http_503", _FakePostResponse(status_code=503, body="overloaded")),
        ]
        for label, primary_outcome in cases:
            for endpoint in ("/v1/chat/completions", "/v1/messages"):
                with self.subTest(case=label, endpoint=endpoint):
                    calls: list[dict[str, Any]] = []
                    outcomes = {
                        "http://127.0.0.1:9001/v1/chat/completions": primary_outcome,
                        "http://127.0.0.1:9002/v1/chat/completions": _FakePostResponse(
                            status_code=200, body=_ok_completion_json("recovered-from-" + label)
                        ),
                    }
                    req = _make_json_request(
                        endpoint,
                        {"model": "local-primary", "messages": [{"role": "user", "content": "ping"}]},
                    )
                    with patch.object(gateway, "MODE", "alfred"), \
                         patch.object(gateway, "MASTER_KEY", "secret"), \
                         patch.object(gateway, "BACKENDS", {"local-primary": primary, "local-backup": backup}), \
                         patch.object(gateway, "FALLBACKS", {"local-primary": ["local-backup"]}), \
                         patch.object(gateway, "TRACING_ENABLED", False), \
                         patch.object(gateway, "verify_swap_in_loaded", AsyncMock(return_value=True)), \
                         patch.object(gateway.httpx, "AsyncClient", _make_fake_httpx_client(outcomes, calls)):
                        if endpoint == "/v1/chat/completions":
                            resp = await gateway.chat_completions(req, authorization="Bearer secret")
                        else:
                            resp = await gateway.messages(req, authorization="Bearer secret", x_api_key=None)
                    self.assertEqual(200, resp.status_code)
                    self.assertEqual(2, len(calls))

    async def test_stream_non_sse_garbage_does_not_leak_and_returns_typed_502_when_exhausted(self):
        primary = gateway.Backend("local-primary", "http://127.0.0.1:9001/v1", "p-model", "k")
        calls: list[dict[str, Any]] = []
        outcomes = {
            "http://127.0.0.1:9001/v1/chat/completions": _FakeStreamResponse(
                status_code=200,
                headers={"content-type": "text/html"},
                body=b"<html>proxy error</html>",
            )
        }
        req = _make_json_request(
            "/v1/chat/completions",
            {"model": "local-primary", "stream": True, "messages": [{"role": "user", "content": "hi"}]},
        )
        with patch.object(gateway, "MODE", "alfred"), \
             patch.object(gateway, "MASTER_KEY", "secret"), \
             patch.object(gateway, "BACKENDS", {"local-primary": primary}), \
             patch.object(gateway, "FALLBACKS", {}), \
             patch.object(gateway, "TRACING_ENABLED", False), \
             patch.object(gateway, "verify_swap_in_loaded", AsyncMock(return_value=True)), \
             patch.object(gateway.httpx, "AsyncClient", _make_fake_httpx_client(outcomes, calls)):
            resp = await gateway.chat_completions(req, authorization="Bearer secret")
        self.assertEqual(502, resp.status_code)
        payload = json.loads(resp.body)
        self.assertEqual("all_backends_failed", payload["error"]["code"])
        self.assertNotIn(b"<html>", resp.body)


# ===========================================================================
# GW-04: Buffered-to-stream adapter preserves reasoning_content identically
# ===========================================================================
@unittest.skipUnless(_HAS_WEB_DEPS, "fastapi and httpx required")
class NormalizedCompletionParityTests(unittest.IsolatedAsyncioTestCase):
    async def test_buffered_json_and_fake_sse_produce_identical_anthropic_blocks(self):
        fixture = {
            "id": "chatcmpl-norm",
            "object": "chat.completion",
            "created": 1700000001,
            "model": "upstream-model",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "reasoning_content": "checking ticker symbol first",
                    "content": "Looking that up now.",
                    "tool_calls": [{
                        "id": "call_123",
                        "type": "function",
                        "function": {"name": "get_quote", "arguments": '{"ticker":"NVDA"}'},
                    }],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 42, "completion_tokens": 19, "total_tokens": 61},
        }
        json_translated = gateway.openai_to_anthropic_response(fixture, "alfred-model")
        fake_sse = gateway._json_to_fake_sse(fixture)
        stream_out = await _collect_stream(
            gateway.openai_stream_to_anthropic(_chunked_iter(fake_sse, 11), "alfred-model")
        )
        stream_reconstructed = _reconstruct_anthropic_from_sse(stream_out)

        self.assertTrue(stream_reconstructed["_saw_message_stop"])
        self.assertEqual(json_translated["content"], stream_reconstructed["content"])
        self.assertEqual(json_translated["stop_reason"], stream_reconstructed["stop_reason"])
        self.assertEqual(json_translated["usage"], stream_reconstructed["usage"])
        self.assertEqual("thinking", json_translated["content"][0]["type"])
        self.assertEqual("checking ticker symbol first", json_translated["content"][0]["thinking"])


# ===========================================================================
# GW-05: Anthropic request translation & Walter tool_choice preservation
# ===========================================================================
@unittest.skipUnless(_HAS_WEB_DEPS, "fastapi and httpx required")
class ProtocolTranslationAndControlsTests(unittest.IsolatedAsyncioTestCase):
    def test_anthropic_tool_choice_none_and_parallel_control_preserved(self):
        out = gateway.anthropic_to_openai_request({
            "model": "m",
            "messages": [{"role": "user", "content": "answer directly"}],
            "tools": [{"name": "search", "description": "s", "input_schema": {"type": "object"}}],
            "tool_choice": {"type": "none", "disable_parallel_tool_use": True},
        })
        self.assertEqual("none", out["tool_choice"])
        self.assertIs(False, out["parallel_tool_calls"])

    def test_anthropic_images_base64_and_url_are_preserved(self):
        out = gateway.anthropic_to_openai_request({
            "model": "m",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe both"},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw0KGgo="}},
                    {"type": "image", "source": {"type": "url", "url": "https://example.com/chart.png"}},
                ],
            }],
        })
        user_content = out["messages"][0]["content"]
        self.assertIsInstance(user_content, list)
        self.assertEqual(
            [
                {"type": "text", "text": "describe both"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}},
                {"type": "image_url", "image_url": {"url": "https://example.com/chart.png"}},
            ],
            user_content,
        )

    async def test_anthropic_unsupported_image_or_tool_choice_returns_typed_400(self):
        bad_payloads = [
            {
                "model": "local-primary",
                "messages": [{"role": "user", "content": [{"type": "image", "source": {"type": "file", "file_id": "f1"}}]}],
            },
            {
                "model": "local-primary",
                "messages": [{"role": "user", "content": [{"type": "document", "source": {"type": "text", "data": "x"}}]}],
            },
            {
                "model": "local-primary",
                "messages": [{"role": "user", "content": "hi"}],
                "tool_choice": {"type": "unsupported_mode"},
            },
        ]
        backend = gateway.Backend("local-primary", "http://127.0.0.1:9001/v1", "p", "k")
        for payload in bad_payloads:
            with self.subTest(payload=payload):
                req = _make_json_request("/v1/messages", payload)
                with patch.object(gateway, "MODE", "alfred"), \
                     patch.object(gateway, "MASTER_KEY", "secret"), \
                     patch.object(gateway, "BACKENDS", {"local-primary": backend}):
                    resp = await gateway.messages(req, authorization="Bearer secret", x_api_key=None)
                self.assertEqual(400, resp.status_code)
                body = json.loads(resp.body)
                self.assertEqual("error", body["type"])
                self.assertEqual("invalid_request_error", body["error"]["type"])

    async def test_walter_honors_tool_choice_none_and_rejects_unsupported_forced_choice(self):
        gateway._session_states.clear()
        gateway._inflight_per_backend.clear()
        tools = [{"type": "function", "function": {"name": "lookup", "description": "d", "parameters": {}}}]

        no_tools_mock = AsyncMock(return_value="no-tools-path-used")
        tools_mock = AsyncMock(return_value="tools-path-used")
        req_none = _make_json_request(
            "/v1/chat/completions",
            {
                "model": "qwen35",
                "messages": [{"role": "user", "content": "just reply"}],
                "tools": tools,
                "tool_choice": "none",
            },
            extra_headers=[(b"x-gordon-session-id", b"tc-none-session")],
        )
        with patch.object(gateway, "_walter_no_tools_path", no_tools_mock), \
             patch.object(gateway, "_walter_tools_path", tools_mock), \
             patch.object(gateway, "get_memory_warning", return_value=(None, 64.0)), \
             patch.object(gateway, "bolog"):
            res = await gateway.walter_handle(req_none)
        self.assertEqual("no-tools-path-used", res)
        tools_mock.assert_not_awaited()
        no_tools_mock.assert_awaited_once()

        req_required = _make_json_request(
            "/v1/chat/completions",
            {
                "model": "qwen35",
                "messages": [{"role": "user", "content": "force tool"}],
                "tools": tools,
                "tool_choice": "required",
            },
            extra_headers=[(b"x-gordon-session-id", b"tc-req-session")],
        )
        with patch.object(gateway, "bolog"):
            resp_bad = await gateway.walter_handle(req_required)
        self.assertEqual(400, resp_bad.status_code)
        self.assertEqual("unsupported_tool_choice", json.loads(resp_bad.body)["error"]["code"])


# ===========================================================================
# GW-06 & GW-07: Telemetry defaults, OTLP TLS enforcement, secret contract, generic defaults
# ===========================================================================
class TelemetryAndHygieneTests(unittest.TestCase):
    def test_example_config_and_packaging_agree_on_secrets_and_defaults(self):
        cfg_text = (ROOT / "config.example.yaml").read_text()
        self.assertIn("trace_content: false", cfg_text)
        self.assertNotIn("from env / the launchd plist", cfg_text)
        self.assertNotIn("keep in plist", cfg_text)
        self.assertIn("GATEWAY_SECRET_ENV", cfg_text)
        self.assertTrue((ROOT / "requirements.txt").exists())
        self.assertTrue((ROOT / "pyproject.toml").exists())

        # No private deployment identifiers in tracked config/code defaults
        for rel in ("gateway.py", "config.example.yaml"):
            text = (ROOT / rel).read_text()
            self.assertNotRegex(text, r"\b100\.(6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.\d+\.\d+\b")
            self.assertNotRegex(text, r"/Users/[A-Za-z0-9._-]+/")
            self.assertNotRegex(text, r"\bU0[A-Z0-9]{8,}\b")

    @unittest.skipUnless(_HAS_WEB_DEPS, "fastapi and httpx required")
    def test_telemetry_defaults_metadata_only_and_enforces_authenticated_tls_outside_loopback(self):
        # Default TRACE_CONTENT must be False (metadata-only)
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TRACE_CONTENT", None)
            self.assertFalse(gateway._get_bool("trace_content", "TRACE_CONTENT", False))

        # Loopback HTTP allowed
        insecure, headers = gateway._validate_otel_transport("http://127.0.0.1:4317", "")
        self.assertTrue(insecure)
        self.assertEqual({}, headers)

        # Loopback HTTPS uses TLS
        insecure, headers = gateway._validate_otel_transport("https://localhost:4317", "")
        self.assertFalse(insecure)

        # Remote plaintext rejected
        with self.assertRaises(ValueError):
            gateway._validate_otel_transport("http://collector.example.com:4317", "authorization=Bearer tok")

        # Remote HTTPS without auth header rejected
        with self.assertRaises(ValueError):
            gateway._validate_otel_transport("https://collector.example.com:4317", "")

        # Remote HTTPS with auth header accepted with insecure=False
        insecure, headers = gateway._validate_otel_transport(
            "https://collector.example.com:4317",
            "authorization=Bearer demo-token",
        )
        self.assertFalse(insecure)
        self.assertEqual({"authorization": "Bearer demo-token"}, headers)

    @unittest.skipUnless(_HAS_WEB_DEPS, "fastapi and httpx required")
    def test_default_backends_and_fallbacks_are_generic_and_local_safe(self):
        defaults = gateway._default_alfred_backends()
        for name, backend in defaults.items():
            self.assertNotIn("/Users/", backend.upstream_model, f"backend {name} leaks user path")
            if name != "openrouter-free":
                self.assertEqual("local", backend.trust_boundary, f"default backend {name} must default to loopback")
            for v in backend.extra_headers.values():
                self.assertNotRegex(v, r"\b100\.(6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.\d+\.\d+\b")
        self.assertEqual("external", defaults["openrouter-free"].trust_boundary)
        self.assertEqual("local", defaults["qwen-397b"].trust_boundary)


if __name__ == "__main__":
    unittest.main()

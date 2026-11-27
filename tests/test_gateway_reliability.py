"""Comprehensive offline regression tests for GW-01 through GW-09.

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
  - GW-08: Separation of usage/metadata-only SSE frames from real completion
    evidence and one consistent post-`[DONE]` terminal-marker policy across
    both OpenAI and Anthropic routes (exercised end-to-end via ASGI).
  - GW-09: Legacy OpenAI `function_call` translation (streaming and non-streaming)
    into Anthropic `tool_use` content blocks with deterministic IDs alongside
    modern `tool_calls` control (exercised end-to-end via ASGI).
"""

from __future__ import annotations

import gzip
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, AsyncIterator
import unittest
from unittest.mock import AsyncMock, patch
import zlib


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

    async def aiter_bytes(self) -> AsyncIterator[bytes]:
        for c in self._chunks:
            yield c
        if self._midstream_exc is not None:
            raise self._midstream_exc

    async def aclose(self) -> None:
        self.closed = True


class _RecordingSpan:
    def __init__(self) -> None:
        self.attributes: dict[str, Any] = {}
        self.ended: bool = False

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def end(self) -> None:
        self.ended = True


class _ChunkedAsyncByteStream(httpx.AsyncByteStream if _HAS_WEB_DEPS else object):
    def __init__(self, payload: bytes, chunk_size: int = 11) -> None:
        self._payload = payload
        self._chunk_size = chunk_size

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for i in range(0, len(self._payload), self._chunk_size):
            yield self._payload[i:i + self._chunk_size]

    async def aclose(self) -> None:
        return None


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

    async def test_upstream_error_event_before_done_propagates_error_and_marks_span(self):
        cases = [
            (
                "explicit_event_error_only",
                b'event: error\r\ndata: {"error":{"type":"overloaded_error","message":"backend overloaded"}}\r\n\r\n'
                b"data: [DONE]\r\n\r\n",
                "backend overloaded",
                "overloaded_error",
            ),
            (
                "json_error_payload_only",
                b'data: {"error":{"type":"rate_limit_error","message":"quota exceeded"}}\n\n'
                b"data: [DONE]\n\n",
                "quota exceeded",
                "rate_limit_error",
            ),
            (
                "midstream_event_error_then_done",
                b'data: {"id":"c1","model":"up","choices":[{"index":0,"delta":{"content":"partial "},"finish_reason":null}]}\r\n\r\n'
                b'event: error\r\ndata: {"error":{"type":"server_error","message":"midstream failure"}}\r\n\r\n'
                b"data: [DONE]\r\n\r\n",
                "midstream failure",
                "server_error",
            ),
            (
                "plain_text_event_error_then_done",
                b"event: error\ndata: upstream crashed hard\n\n"
                b"data: [DONE]\n\n",
                "upstream crashed hard",
                None,
            ),
        ]
        for label, raw_sse, expected_msg, expected_type in cases:
            with self.subTest(case=label):
                # Direct translator checks: Anthropic
                out_a = await _collect_stream(
                    gateway.openai_stream_to_anthropic(_chunked_iter(raw_sse, 7), "req-model")
                )
                events_a = _parse_output_sse(out_a)
                etypes_a = [et for et, _ in events_a]
                self.assertEqual("error", etypes_a[-1])
                self.assertNotIn("message_stop", etypes_a)
                self.assertNotIn("message_delta", etypes_a)
                self.assertNotIn(b"end_turn", out_a)
                err_payload_a = json.loads(events_a[-1][1])
                self.assertEqual("error", err_payload_a["type"])
                self.assertIn(expected_msg, err_payload_a["error"]["message"])
                if expected_type:
                    self.assertEqual(expected_type, err_payload_a["error"]["type"])

                # Direct translator checks: OpenAI
                out_o = await _collect_stream(
                    gateway.rewrite_stream_model(_chunked_iter(raw_sse, 7), "req-model")
                )
                events_o = _parse_output_sse(out_o)
                etypes_o = [et for et, _ in events_o]
                self.assertEqual("error", etypes_o[-1])
                self.assertNotIn(b"[DONE]", out_o)
                err_payload_o = json.loads(events_o[-1][1])
                self.assertIn(expected_msg, err_payload_o["error"]["message"])
                if expected_type:
                    self.assertEqual(expected_type, err_payload_o["error"]["type"])

                # End-to-end route + span verification (both /v1/messages and /v1/chat/completions)
                backend = gateway.Backend("local-primary", "http://127.0.0.1:9001/v1", "up", "k")
                for endpoint in ("/v1/messages", "/v1/chat/completions"):
                    calls: list[dict[str, Any]] = []
                    outcomes = {
                        "http://127.0.0.1:9001/v1/chat/completions": _FakeStreamResponse(
                            status_code=200,
                            chunks=[raw_sse[:19], raw_sse[19:]],
                        )
                    }
                    span = _RecordingSpan()
                    req = _make_json_request(
                        endpoint,
                        {"model": "local-primary", "stream": True, "messages": [{"role": "user", "content": "hi"}]},
                    )
                    with patch.object(gateway, "MODE", "alfred"), \
                         patch.object(gateway, "MASTER_KEY", "secret"), \
                         patch.object(gateway, "BACKENDS", {"local-primary": backend}), \
                         patch.object(gateway, "FALLBACKS", {}), \
                         patch.object(gateway, "TRACING_ENABLED", True), \
                         patch.object(gateway, "_start_chat_span", return_value=span), \
                         patch.object(gateway, "_record_metrics"), \
                         patch.object(gateway, "verify_swap_in_loaded", AsyncMock(return_value=True)), \
                         patch.object(gateway.httpx, "AsyncClient", _make_fake_httpx_client(outcomes, calls)):
                        if endpoint == "/v1/messages":
                            resp = await gateway.messages(req, authorization="Bearer secret", x_api_key=None)
                        else:
                            resp = await gateway.chat_completions(req, authorization="Bearer secret")
                        self.assertEqual(200, resp.status_code)
                        body = await _collect_stream(resp.body_iterator)
                    events = _parse_output_sse(body)
                    self.assertEqual("error", events[-1][0])
                    self.assertIn(expected_msg.encode("utf-8"), body)
                    self.assertNotIn(b"message_stop", body)
                    self.assertNotIn(b"end_turn", body)
                    self.assertNotIn(b"[DONE]", body)
                    self.assertIs(True, span.attributes.get("error"), f"span must mark error=True on {endpoint}/{label}")
                    self.assertTrue(span.ended)

    async def test_unrecognized_sse_schema_before_done_never_reports_success(self):
        raw = b'data: {"status":"unexpected_schema"}\n\ndata: [DONE]\n\n'
        out_a = await _collect_stream(gateway.openai_stream_to_anthropic(_chunked_iter(raw, 5), "m"))
        etypes_a = [et for et, _ in _parse_output_sse(out_a)]
        self.assertEqual("error", etypes_a[-1])
        self.assertNotIn("message_stop", etypes_a)
        self.assertNotIn(b"end_turn", out_a)

        out_o = await _collect_stream(gateway.rewrite_stream_model(_chunked_iter(raw, 5), "m"))
        etypes_o = [et for et, _ in _parse_output_sse(out_o)]
        self.assertEqual("error", etypes_o[-1])
        self.assertNotIn(b"[DONE]", out_o)

    async def test_compressed_gzip_and_deflate_sse_decodes_cleanly_via_real_httpx_transport(self):
        raw_sse = self._build_full_sse(b"\r\n")
        encodings = [
            ("gzip", gzip.compress(raw_sse)),
            ("deflate", zlib.compress(raw_sse)),
        ]
        backend = gateway.Backend("local-primary", "http://127.0.0.1:9001/v1", "upstream-model", "k")

        for encoding, compressed_bytes in encodings:
            for endpoint in ("/v1/messages", "/v1/chat/completions"):
                with self.subTest(content_encoding=encoding, endpoint=endpoint):
                    def make_handler(enc: str, payload: bytes):
                        def _handler(_request: httpx.Request) -> httpx.Response:
                            return httpx.Response(
                                status_code=200,
                                headers={
                                    "content-type": "text/event-stream",
                                    "content-encoding": enc,
                                },
                                stream=_ChunkedAsyncByteStream(payload, chunk_size=13),
                            )
                        return _handler

                    transport = httpx.MockTransport(make_handler(encoding, compressed_bytes))
                    real_async_client = httpx.AsyncClient

                    def _client_factory(*args, **kwargs):
                        kwargs["transport"] = transport
                        return real_async_client(*args, **kwargs)

                    req = _make_json_request(
                        endpoint,
                        {"model": "local-primary", "stream": True, "messages": [{"role": "user", "content": "hi"}]},
                    )
                    with patch.object(gateway, "MODE", "alfred"), \
                         patch.object(gateway, "MASTER_KEY", "secret"), \
                         patch.object(gateway, "BACKENDS", {"local-primary": backend}), \
                         patch.object(gateway, "FALLBACKS", {}), \
                         patch.object(gateway, "TRACING_ENABLED", False), \
                         patch.object(gateway, "verify_swap_in_loaded", AsyncMock(return_value=True)), \
                         patch.object(gateway.httpx, "AsyncClient", side_effect=_client_factory):
                        if endpoint == "/v1/messages":
                            resp = await gateway.messages(req, authorization="Bearer secret", x_api_key=None)
                        else:
                            resp = await gateway.chat_completions(req, authorization="Bearer secret")
                        self.assertEqual(200, resp.status_code)
                        body = await _collect_stream(resp.body_iterator)

                    if endpoint == "/v1/messages":
                        reconstructed = _reconstruct_anthropic_from_sse(body)
                        self.assertTrue(reconstructed["_saw_message_stop"])
                        self.assertEqual("tool_use", reconstructed["stop_reason"])
                        self.assertEqual(
                            [
                                {"type": "thinking", "thinking": "step 1"},
                                {"type": "text", "text": "hello world"},
                                {"type": "tool_use", "id": "call_abc", "name": "lookup", "input": {"sym": "AAPL"}},
                            ],
                            reconstructed["content"],
                        )
                    else:
                        events = _parse_output_sse(body)
                        self.assertEqual("[DONE]", events[-1][1])
                        self.assertNotIn("error", [et for et, _ in events])
                        objs = [json.loads(d) for _, d in events[:-1]]
                        self.assertTrue(all(o["model"] == "local-primary" for o in objs))
                        self.assertEqual(
                            "hello world",
                            "".join(o["choices"][0]["delta"].get("content", "") for o in objs),
                        )

    async def test_corrupt_gzip_sse_fails_over_before_http_200_commit(self):
        primary = gateway.Backend("local-primary", "http://127.0.0.1:9001/v1", "p-model", "k")
        backup = gateway.Backend("local-backup", "http://127.0.0.1:9002/v1", "b-model", "k")
        good_gzip = gzip.compress(
            b'data: {"id":"c2","model":"b-model","choices":[{"index":0,"delta":{"content":"decoded-backup"},"finish_reason":"stop"}]}\r\n\r\n'
            b"data: [DONE]\r\n\r\n"
        )
        seen_urls: list[str] = []

        def _handler(request: httpx.Request) -> httpx.Response:
            seen_urls.append(str(request.url))
            if "9001" in str(request.url):
                return httpx.Response(
                    status_code=200,
                    headers={"content-type": "text/event-stream", "content-encoding": "gzip"},
                    stream=_ChunkedAsyncByteStream(b"not-valid-gzip-wire-bytes", chunk_size=7),
                )
            return httpx.Response(
                status_code=200,
                headers={"content-type": "text/event-stream", "content-encoding": "gzip"},
                stream=_ChunkedAsyncByteStream(good_gzip, chunk_size=9),
            )

        transport = httpx.MockTransport(_handler)
        real_async_client = httpx.AsyncClient

        def _client_factory(*args, **kwargs):
            kwargs["transport"] = transport
            return real_async_client(*args, **kwargs)

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
             patch.object(gateway.httpx, "AsyncClient", side_effect=_client_factory):
            resp = await gateway.messages(req, authorization="Bearer secret", x_api_key=None)
            self.assertEqual(200, resp.status_code)
            body = await _collect_stream(resp.body_iterator)

        self.assertEqual(2, len(seen_urls))
        reconstructed = _reconstruct_anthropic_from_sse(body)
        self.assertTrue(reconstructed["_saw_message_stop"])
        self.assertEqual([{"type": "text", "text": "decoded-backup"}], reconstructed["content"])

    async def test_usage_only_or_metadata_only_stream_emits_error_never_success_in_translators(self):
        cases = [
            (
                "usage_empty_choices_then_done",
                b'data: {"id":"c1","model":"up","choices":[],"usage":{"prompt_tokens":12,"completion_tokens":0}}\n\n'
                b"data: [DONE]\n\n",
            ),
            (
                "usage_without_choices_then_done",
                b'data: {"id":"c1","model":"up","usage":{"prompt_tokens":12,"completion_tokens":0}}\r\n\r\n'
                b"data: [DONE]\r\n\r\n",
            ),
            (
                "usage_empty_choices_eof",
                b'data: {"id":"c1","model":"up","choices":[],"usage":{"prompt_tokens":12,"completion_tokens":0}}\n\n',
            ),
            (
                "role_only_empty_delta_then_done",
                b'data: {"id":"c1","model":"up","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}\n\n'
                b"data: [DONE]\n\n",
            ),
        ]
        for label, raw_sse in cases:
            for chunk_size in (7, len(raw_sse)):
                with self.subTest(case=label, chunk_size=chunk_size):
                    out_o = await _collect_stream(
                        gateway.rewrite_stream_model(_chunked_iter(raw_sse, chunk_size), "req-model")
                    )
                    events_o = _parse_output_sse(out_o)
                    self.assertEqual(["error"], [et for et, _ in events_o])
                    self.assertNotIn(b"[DONE]", out_o)
                    err_o = json.loads(events_o[0][1])
                    self.assertEqual("upstream_stream_error", err_o["error"]["type"])

                    out_a = await _collect_stream(
                        gateway.openai_stream_to_anthropic(_chunked_iter(raw_sse, chunk_size), "req-model")
                    )
                    events_a = _parse_output_sse(out_a)
                    self.assertEqual(["error"], [et for et, _ in events_a])
                    self.assertNotIn(b"message_start", out_a)
                    self.assertNotIn(b"message_delta", out_a)
                    self.assertNotIn(b"message_stop", out_a)
                    err_a = json.loads(events_a[0][1])
                    self.assertEqual("error", err_a["type"])
                    self.assertEqual("api_error", err_a["error"]["type"])

    async def test_post_done_trailing_error_and_garbage_ignored_consistently_in_translators(self):
        base = (
            b'data: {"id":"c1","model":"up","choices":[{"index":0,"delta":{"content":"hello"},"finish_reason":"stop"}],'
            b'"usage":{"prompt_tokens":9,"completion_tokens":3}}\n\n'
            b"data: [DONE]\n\n"
        )
        trailing_cases = [
            (
                "post_done_event_error",
                base + b'event: error\ndata: {"error":{"type":"server_error","message":"late diagnostic"}}\n\n',
            ),
            (
                "post_done_json_error",
                base + b'data: {"error":{"type":"server_error","message":"late json diagnostic"}}\n\n',
            ),
            (
                "post_done_unterminated_garbage",
                base + b'data: {"truncated_after_done',
            ),
        ]
        for label, raw_sse in trailing_cases:
            for chunk_size in (11, len(raw_sse)):
                with self.subTest(case=label, chunk_size=chunk_size):
                    out_o = await _collect_stream(
                        gateway.rewrite_stream_model(_chunked_iter(raw_sse, chunk_size), "req-model")
                    )
                    events_o = _parse_output_sse(out_o)
                    self.assertEqual("[DONE]", events_o[-1][1])
                    self.assertNotIn("error", [et for et, _ in events_o])
                    self.assertNotIn(b"late diagnostic", out_o)
                    self.assertNotIn(b"late json diagnostic", out_o)

                    out_a = await _collect_stream(
                        gateway.openai_stream_to_anthropic(_chunked_iter(raw_sse, chunk_size), "req-model")
                    )
                    events_a = _parse_output_sse(out_a)
                    self.assertNotIn("error", [et for et, _ in events_a])
                    reconstructed = _reconstruct_anthropic_from_sse(out_a)
                    self.assertTrue(reconstructed["_saw_message_stop"])
                    self.assertEqual("end_turn", reconstructed["stop_reason"])
                    self.assertEqual([{"type": "text", "text": "hello"}], reconstructed["content"])
                    self.assertEqual({"input_tokens": 9, "output_tokens": 3}, reconstructed["usage"])

    async def test_leading_metadata_and_trailing_usage_preserved_with_valid_completion(self):
        raw_sse = (
            b'data: {"id":"c1","model":"up","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}\n\n'
            b'data: {"id":"c1","model":"up","choices":[{"index":0,"delta":{"content":"hi there"},"finish_reason":"stop"}]}\n\n'
            b'data: {"id":"c1","model":"up","choices":[],"usage":{"prompt_tokens":14,"completion_tokens":6}}\n\n'
            b"data: [DONE]\n\n"
        )
        out_o = await _collect_stream(gateway.rewrite_stream_model(_chunked_iter(raw_sse, 13), "req-model"))
        events_o = _parse_output_sse(out_o)
        self.assertEqual("[DONE]", events_o[-1][1])
        self.assertNotIn("error", [et for et, _ in events_o])
        objs_o = [json.loads(d) for _, d in events_o[:-1]]
        self.assertEqual(3, len(objs_o))
        self.assertTrue(all(o["model"] == "req-model" for o in objs_o))
        self.assertEqual({"prompt_tokens": 14, "completion_tokens": 6}, objs_o[-1]["usage"])

        out_a = await _collect_stream(gateway.openai_stream_to_anthropic(_chunked_iter(raw_sse, 13), "req-model"))
        reconstructed = _reconstruct_anthropic_from_sse(out_a)
        self.assertTrue(reconstructed["_saw_message_stop"])
        self.assertEqual([{"type": "text", "text": "hi there"}], reconstructed["content"])
        self.assertEqual({"input_tokens": 14, "output_tokens": 6}, reconstructed["usage"])

    async def test_asgi_usage_only_stream_rejected_and_fails_over_on_both_routes(self):
        usage_only_done = (
            b'data: {"id":"c1","model":"p-model","choices":[],"usage":{"prompt_tokens":15,"completion_tokens":0}}\n\n'
            b"data: [DONE]\n\n"
        )
        valid_backup = (
            b'data: {"id":"c2","model":"b-model","choices":[{"index":0,"delta":{"content":"from-backup"},"finish_reason":"stop"}],'
            b'"usage":{"prompt_tokens":15,"completion_tokens":4}}\n\n'
            b"data: [DONE]\n\n"
        )
        primary = gateway.Backend("local-primary", "http://127.0.0.1:9001/v1", "p-model", "k")
        backup = gateway.Backend("local-backup", "http://127.0.0.1:9002/v1", "b-model", "k")
        auth = {"Authorization": "Bearer secret"}

        # 1. Single backend returning usage-only + [DONE] is rejected as an error on BOTH routes.
        for endpoint in ("/v1/chat/completions", "/v1/messages"):
            for chunk_size in (17, len(usage_only_done)):
                with self.subTest(mode="single_backend", endpoint=endpoint, chunk_size=chunk_size):
                    def _single_handler(_request: httpx.Request) -> httpx.Response:
                        return httpx.Response(
                            status_code=200,
                            headers={"content-type": "text/event-stream"},
                            stream=_ChunkedAsyncByteStream(usage_only_done, chunk_size=chunk_size),
                        )

                    upstream_transport = httpx.MockTransport(_single_handler)
                    real_async_client = httpx.AsyncClient

                    def _client_factory(*args, **kwargs):
                        kwargs["transport"] = upstream_transport
                        return real_async_client(*args, **kwargs)

                    asgi_transport = httpx.ASGITransport(app=gateway.app)
                    async with real_async_client(transport=asgi_transport, base_url="http://gateway.local") as client:
                        with patch.object(gateway, "MODE", "alfred"), \
                             patch.object(gateway, "MASTER_KEY", "secret"), \
                             patch.object(gateway, "BACKENDS", {"local-primary": primary}), \
                             patch.object(gateway, "FALLBACKS", {}), \
                             patch.object(gateway, "TRACING_ENABLED", False), \
                             patch.object(gateway, "verify_swap_in_loaded", AsyncMock(return_value=True)), \
                             patch.object(gateway.httpx, "AsyncClient", side_effect=_client_factory):
                            resp = await client.post(
                                endpoint,
                                headers=auth,
                                json={"model": "local-primary", "stream": True, "messages": [{"role": "user", "content": "hi"}]},
                            )
                    self.assertEqual(502, resp.status_code)
                    self.assertNotIn(b"[DONE]", resp.content)
                    self.assertNotIn(b"message_start", resp.content)
                    self.assertNotIn(b"message_delta", resp.content)
                    self.assertNotIn(b"message_stop", resp.content)
                    err_json = resp.json()
                    self.assertEqual("all_backends_failed", err_json["error"]["code"])
                    self.assertEqual("upstream_protocol_error", err_json["error"]["attempts"][0]["error_type"])

        # 2. With a fallback backend configured, usage-only + [DONE] fails over cleanly before HTTP 200.
        for endpoint in ("/v1/chat/completions", "/v1/messages"):
            with self.subTest(mode="failover", endpoint=endpoint):
                seen_urls: list[str] = []

                def _failover_handler(request: httpx.Request) -> httpx.Response:
                    seen_urls.append(str(request.url))
                    payload = usage_only_done if "9001" in str(request.url) else valid_backup
                    return httpx.Response(
                        status_code=200,
                        headers={"content-type": "text/event-stream"},
                        stream=_ChunkedAsyncByteStream(payload, chunk_size=19),
                    )

                upstream_transport = httpx.MockTransport(_failover_handler)
                real_async_client = httpx.AsyncClient

                def _client_factory(*args, **kwargs):
                    kwargs["transport"] = upstream_transport
                    return real_async_client(*args, **kwargs)

                asgi_transport = httpx.ASGITransport(app=gateway.app)
                async with real_async_client(transport=asgi_transport, base_url="http://gateway.local") as client:
                    with patch.object(gateway, "MODE", "alfred"), \
                         patch.object(gateway, "MASTER_KEY", "secret"), \
                         patch.object(gateway, "BACKENDS", {"local-primary": primary, "local-backup": backup}), \
                         patch.object(gateway, "FALLBACKS", {"local-primary": ["local-backup"]}), \
                         patch.object(gateway, "TRACING_ENABLED", False), \
                         patch.object(gateway, "verify_swap_in_loaded", AsyncMock(return_value=True)), \
                         patch.object(gateway.httpx, "AsyncClient", side_effect=_client_factory):
                        resp = await client.post(
                            endpoint,
                            headers=auth,
                            json={"model": "local-primary", "stream": True, "messages": [{"role": "user", "content": "hi"}]},
                        )
                self.assertEqual(200, resp.status_code)
                self.assertEqual(2, len(seen_urls))
                if endpoint == "/v1/messages":
                    reconstructed = _reconstruct_anthropic_from_sse(resp.content)
                    self.assertTrue(reconstructed["_saw_message_stop"])
                    self.assertEqual([{"type": "text", "text": "from-backup"}], reconstructed["content"])
                else:
                    events = _parse_output_sse(resp.content)
                    self.assertEqual("[DONE]", events[-1][1])
                    self.assertNotIn("error", [et for et, _ in events])
                    objs = [json.loads(d) for _, d in events[:-1]]
                    self.assertEqual("from-backup", "".join(o["choices"][0]["delta"].get("content", "") for o in objs))

    async def test_asgi_post_done_error_and_controls_consistent_across_openai_and_anthropic(self):
        post_done_error_sse = (
            b'data: {"id":"c1","model":"up","choices":[{"index":0,"delta":{"content":"completed ok"},"finish_reason":"stop"}]}\n\n'
            b'data: {"id":"c1","model":"up","choices":[],"usage":{"prompt_tokens":11,"completion_tokens":5}}\n\n'
            b"data: [DONE]\n\n"
            b'event: error\ndata: {"error":{"type":"server_error","message":"diagnostic after done"}}\n\n'
        )
        normal_with_usage_sse = (
            b'data: {"id":"c2","model":"up","choices":[{"index":0,"delta":{"content":"ordinary ok"},"finish_reason":"stop"}]}\n\n'
            b'data: {"id":"c2","model":"up","choices":[],"usage":{"prompt_tokens":7,"completion_tokens":3}}\n\n'
            b"data: [DONE]\n\n"
        )
        error_before_done_sse = (
            b'data: {"id":"c3","model":"up","choices":[{"index":0,"delta":{"content":"partial"},"finish_reason":null}]}\n\n'
            b'event: error\ndata: {"error":{"type":"server_error","message":"failed before done"}}\n\n'
            b"data: [DONE]\n\n"
        )
        backend = gateway.Backend("local-primary", "http://127.0.0.1:9001/v1", "up", "k")
        auth = {"Authorization": "Bearer secret"}

        # 1. Post-[DONE] diagnostic error: BOTH routes stop cleanly at [DONE] and ignore trailing error.
        for endpoint in ("/v1/chat/completions", "/v1/messages"):
            for chunk_size in (23, len(post_done_error_sse)):
                with self.subTest(case="post_done_error", endpoint=endpoint, chunk_size=chunk_size):
                    closed_flag = {"closed": False}

                    class _TrackedStream(_ChunkedAsyncByteStream):
                        async def aclose(self) -> None:
                            closed_flag["closed"] = True

                    def _handler(_request: httpx.Request) -> httpx.Response:
                        return httpx.Response(
                            status_code=200,
                            headers={"content-type": "text/event-stream"},
                            stream=_TrackedStream(post_done_error_sse, chunk_size=chunk_size),
                        )

                    span = _RecordingSpan()
                    upstream_transport = httpx.MockTransport(_handler)
                    real_async_client = httpx.AsyncClient

                    def _client_factory(*args, **kwargs):
                        kwargs["transport"] = upstream_transport
                        return real_async_client(*args, **kwargs)

                    asgi_transport = httpx.ASGITransport(app=gateway.app)
                    async with real_async_client(transport=asgi_transport, base_url="http://gateway.local") as client:
                        with patch.object(gateway, "MODE", "alfred"), \
                             patch.object(gateway, "MASTER_KEY", "secret"), \
                             patch.object(gateway, "BACKENDS", {"local-primary": backend}), \
                             patch.object(gateway, "FALLBACKS", {}), \
                             patch.object(gateway, "TRACING_ENABLED", True), \
                             patch.object(gateway, "_start_chat_span", return_value=span), \
                             patch.object(gateway, "_record_metrics"), \
                             patch.object(gateway, "verify_swap_in_loaded", AsyncMock(return_value=True)), \
                             patch.object(gateway.httpx, "AsyncClient", side_effect=_client_factory):
                            resp = await client.post(
                                endpoint,
                                headers=auth,
                                json={"model": "local-primary", "stream": True, "messages": [{"role": "user", "content": "hi"}]},
                            )
                    self.assertEqual(200, resp.status_code)
                    self.assertNotIn(b"diagnostic after done", resp.content)
                    self.assertNotIn("error", [et for et, _ in _parse_output_sse(resp.content)])
                    self.assertNotIn("error", span.attributes)
                    self.assertTrue(span.ended)
                    self.assertTrue(closed_flag["closed"], "upstream stream must be closed after stopping at [DONE]")
                    if endpoint == "/v1/messages":
                        reconstructed = _reconstruct_anthropic_from_sse(resp.content)
                        self.assertTrue(reconstructed["_saw_message_stop"])
                        self.assertEqual([{"type": "text", "text": "completed ok"}], reconstructed["content"])
                        self.assertEqual({"input_tokens": 11, "output_tokens": 5}, reconstructed["usage"])
                    else:
                        events = _parse_output_sse(resp.content)
                        self.assertEqual("[DONE]", events[-1][1])
                        objs = [json.loads(d) for _, d in events[:-1]]
                        self.assertEqual("completed ok", objs[0]["choices"][0]["delta"]["content"])
                        self.assertEqual({"prompt_tokens": 11, "completion_tokens": 5}, objs[1]["usage"])

        # 2. Controls: ordinary valid completion + trailing usage passes; error-before-[DONE] fails with error.
        for endpoint in ("/v1/chat/completions", "/v1/messages"):
            with self.subTest(case="ordinary_control", endpoint=endpoint):
                def _ok_handler(_request: httpx.Request) -> httpx.Response:
                    return httpx.Response(
                        status_code=200,
                        headers={"content-type": "text/event-stream"},
                        stream=_ChunkedAsyncByteStream(normal_with_usage_sse, chunk_size=19),
                    )

                upstream_transport = httpx.MockTransport(_ok_handler)
                real_async_client = httpx.AsyncClient

                def _client_factory(*args, **kwargs):
                    kwargs["transport"] = upstream_transport
                    return real_async_client(*args, **kwargs)

                asgi_transport = httpx.ASGITransport(app=gateway.app)
                async with real_async_client(transport=asgi_transport, base_url="http://gateway.local") as client:
                    with patch.object(gateway, "MODE", "alfred"), \
                         patch.object(gateway, "MASTER_KEY", "secret"), \
                         patch.object(gateway, "BACKENDS", {"local-primary": backend}), \
                         patch.object(gateway, "FALLBACKS", {}), \
                         patch.object(gateway, "TRACING_ENABLED", False), \
                         patch.object(gateway, "verify_swap_in_loaded", AsyncMock(return_value=True)), \
                         patch.object(gateway.httpx, "AsyncClient", side_effect=_client_factory):
                        resp = await client.post(
                            endpoint,
                            headers=auth,
                            json={"model": "local-primary", "stream": True, "messages": [{"role": "user", "content": "hi"}]},
                        )
                self.assertEqual(200, resp.status_code)
                if endpoint == "/v1/messages":
                    reconstructed = _reconstruct_anthropic_from_sse(resp.content)
                    self.assertTrue(reconstructed["_saw_message_stop"])
                    self.assertEqual([{"type": "text", "text": "ordinary ok"}], reconstructed["content"])
                    self.assertEqual({"input_tokens": 7, "output_tokens": 3}, reconstructed["usage"])
                else:
                    events = _parse_output_sse(resp.content)
                    self.assertEqual("[DONE]", events[-1][1])
                    self.assertNotIn("error", [et for et, _ in events])

            with self.subTest(case="error_before_done_control", endpoint=endpoint):
                def _err_handler(_request: httpx.Request) -> httpx.Response:
                    return httpx.Response(
                        status_code=200,
                        headers={"content-type": "text/event-stream"},
                        stream=_ChunkedAsyncByteStream(error_before_done_sse, chunk_size=19),
                    )

                span = _RecordingSpan()
                upstream_transport = httpx.MockTransport(_err_handler)
                real_async_client = httpx.AsyncClient

                def _client_factory(*args, **kwargs):
                    kwargs["transport"] = upstream_transport
                    return real_async_client(*args, **kwargs)

                asgi_transport = httpx.ASGITransport(app=gateway.app)
                async with real_async_client(transport=asgi_transport, base_url="http://gateway.local") as client:
                    with patch.object(gateway, "MODE", "alfred"), \
                         patch.object(gateway, "MASTER_KEY", "secret"), \
                         patch.object(gateway, "BACKENDS", {"local-primary": backend}), \
                         patch.object(gateway, "FALLBACKS", {}), \
                         patch.object(gateway, "TRACING_ENABLED", True), \
                         patch.object(gateway, "_start_chat_span", return_value=span), \
                         patch.object(gateway, "_record_metrics"), \
                         patch.object(gateway, "verify_swap_in_loaded", AsyncMock(return_value=True)), \
                         patch.object(gateway.httpx, "AsyncClient", side_effect=_client_factory):
                        resp = await client.post(
                            endpoint,
                            headers=auth,
                            json={"model": "local-primary", "stream": True, "messages": [{"role": "user", "content": "hi"}]},
                        )
                self.assertEqual(200, resp.status_code)
                events = _parse_output_sse(resp.content)
                self.assertEqual("error", events[-1][0])
                self.assertIn(b"failed before done", resp.content)
                self.assertNotIn(b"[DONE]", resp.content)
                self.assertNotIn(b"message_stop", resp.content)
                self.assertNotIn(b"message_delta", resp.content)
                self.assertIs(True, span.attributes.get("error"))
                self.assertTrue(span.ended)


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

    def test_strict_config_rejects_string_booleans_and_normalizes_or_rejects_trust_boundaries(self):
        # 1. Quoted string booleans (e.g. "false") and non-bool types MUST fail closed with ValueError.
        for bad_bool in ("false", "true", "0", "1", 0, 1, None, ["false"]):
            with self.subTest(bad_bool=bad_bool):
                with self.assertRaises(ValueError):
                    gateway.Backend(
                        "backup",
                        "https://external.example/v1",
                        "c-model",
                        "k",
                        trust_boundary="external",
                        allow_external_fallback=bad_bool,  # type: ignore[arg-type]
                    )
                cfg = {
                    "backends": [
                        {
                            "name": "local",
                            "base_url": "http://127.0.0.1:9001/v1",
                            "trust_boundary": "local",
                        },
                        {
                            "name": "backup",
                            "base_url": "https://external.example/v1",
                            "trust_boundary": "external",
                            "allow_external_fallback": bad_bool,
                        },
                    ]
                }
                with patch.object(gateway, "_CFG", cfg):
                    with self.assertRaises(ValueError):
                        gateway._backends_from_config()

        # Top-level YAML allow_external_fallback must also reject string booleans & invalid env values.
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ALLOW_EXTERNAL_FALLBACK", None)
            with patch.object(gateway, "_CFG", {"allow_external_fallback": "false"}):
                with self.assertRaises(ValueError):
                    gateway._get_strict_bool("allow_external_fallback", "ALLOW_EXTERNAL_FALLBACK", False)
        with patch.dict(os.environ, {"ALLOW_EXTERNAL_FALLBACK": "maybe"}, clear=False):
            with self.assertRaises(ValueError):
                gateway._get_strict_bool("allow_external_fallback", "ALLOW_EXTERNAL_FALLBACK", False)
        with patch.dict(os.environ, {"ALLOW_EXTERNAL_FALLBACK": "false"}, clear=False):
            self.assertIs(False, gateway._get_strict_bool("allow_external_fallback", "ALLOW_EXTERNAL_FALLBACK", True))

        # 2. Mixed-case trust_boundary ('LOCAL', 'External') normalizes cleanly and NEVER bypasses guard.
        cfg_mixed = {
            "backends": [
                {
                    "name": "local",
                    "base_url": "http://127.0.0.1:9001/v1",
                    "trust_boundary": "LOCAL",
                    "allow_external_fallback": False,
                },
                {
                    "name": "backup",
                    "base_url": "https://external.example/v1",
                    "trust_boundary": "External",
                    "allow_external_fallback": False,
                },
            ]
        }
        with patch.object(gateway, "_CFG", cfg_mixed):
            parsed_backends = gateway._backends_from_config()
        self.assertEqual("local", parsed_backends["local"].trust_boundary)
        self.assertEqual("external", parsed_backends["backup"].trust_boundary)

        with patch.object(gateway, "BACKENDS", parsed_backends), \
             patch.object(gateway, "FALLBACKS", {"local": ["backup"]}), \
             patch.object(gateway, "ALLOW_EXTERNAL_FALLBACK", False), \
             patch.object(gateway, "EXTERNAL_FALLBACK_ROUTES", frozenset()):
            self.assertEqual(["local"], [b.name for b in gateway.resolve_chain("local")])

        # Defense-in-depth: even post-construction mutation cannot bypass resolve_chain guard.
        parsed_backends["local"].trust_boundary = "LOCAL"
        parsed_backends["backup"].allow_external_fallback = "false"  # type: ignore[assignment]
        with patch.object(gateway, "BACKENDS", parsed_backends), \
             patch.object(gateway, "FALLBACKS", {"local": ["backup"]}), \
             patch.object(gateway, "ALLOW_EXTERNAL_FALLBACK", False), \
             patch.object(gateway, "EXTERNAL_FALLBACK_ROUTES", frozenset()):
            self.assertEqual(["local"], [b.name for b in gateway.resolve_chain("local")])

        # 3. Unknown or non-string trust_boundary labels MUST fail closed with ValueError.
        for bad_boundary in ("dmz", "private", "cloud", "lan", 123, ["local"]):
            with self.subTest(bad_boundary=bad_boundary):
                with self.assertRaises(ValueError):
                    gateway.Backend(
                        "local",
                        "http://127.0.0.1:9001/v1",
                        "p-model",
                        "k",
                        trust_boundary=bad_boundary,  # type: ignore[arg-type]
                    )
                with patch.object(
                    gateway,
                    "_CFG",
                    {"backends": [{"name": "local", "base_url": "http://127.0.0.1:9001/v1", "trust_boundary": bad_boundary}]},
                ):
                    with self.assertRaises(ValueError):
                        gateway._backends_from_config()

        # 4. Typed boolean allow_external_fallback=True still explicitly authorizes external hop.
        cfg_allowed = {
            "backends": [
                {"name": "local", "base_url": "http://127.0.0.1:9001/v1", "trust_boundary": "local"},
                {
                    "name": "backup",
                    "base_url": "https://external.example/v1",
                    "trust_boundary": "external",
                    "allow_external_fallback": True,
                },
            ]
        }
        with patch.object(gateway, "_CFG", cfg_allowed):
            allowed_backends = gateway._backends_from_config()
        with patch.object(gateway, "BACKENDS", allowed_backends), \
             patch.object(gateway, "FALLBACKS", {"local": ["backup"]}), \
             patch.object(gateway, "ALLOW_EXTERNAL_FALLBACK", False), \
             patch.object(gateway, "EXTERNAL_FALLBACK_ROUTES", frozenset()):
            self.assertEqual(["local", "backup"], [b.name for b in gateway.resolve_chain("local")])

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


# ===========================================================================
# GW-09: Legacy OpenAI function_call translation into Anthropic tool_use blocks
# ===========================================================================
@unittest.skipUnless(_HAS_WEB_DEPS, "fastapi and httpx required")
class LegacyFunctionCallTranslationTests(unittest.IsolatedAsyncioTestCase):
    async def test_asgi_streaming_legacy_function_call_translated_to_anthropic_tool_use(self):
        legacy_sse = (
            b'data: {"id":"chatcmpl-legacy-stream","object":"chat.completion.chunk","model":"up",'
            b'"choices":[{"index":0,"delta":{"role":"assistant","content":null,'
            b'"function_call":{"name":"get_weather","arguments":""}},"finish_reason":null}]}\n\n'
            b'data: {"id":"chatcmpl-legacy-stream","object":"chat.completion.chunk","model":"up",'
            b'"choices":[{"index":0,"delta":{"function_call":{"arguments":"{\\"city\\":"}},"finish_reason":null}]}\n\n'
            b'data: {"id":"chatcmpl-legacy-stream","object":"chat.completion.chunk","model":"up",'
            b'"choices":[{"index":0,"delta":{"function_call":{"arguments":"\\"Seattle\\",\\"unit\\":\\"c\\"}"}},'
            b'"finish_reason":null}]}\n\n'
            b'data: {"id":"chatcmpl-legacy-stream","object":"chat.completion.chunk","model":"up",'
            b'"choices":[{"index":0,"delta":{},"finish_reason":"function_call"}],'
            b'"usage":{"prompt_tokens":21,"completion_tokens":9}}\n\n'
            b"data: [DONE]\n\n"
        )
        expected_id = gateway._legacy_function_call_id("chatcmpl-legacy-stream", 0)
        backend = gateway.Backend("local-primary", "http://127.0.0.1:9001/v1", "up", "k")
        auth = {"Authorization": "Bearer secret"}

        for chunk_size in (13, len(legacy_sse)):
            with self.subTest(chunk_size=chunk_size):
                def _handler(_request: httpx.Request) -> httpx.Response:
                    return httpx.Response(
                        status_code=200,
                        headers={"content-type": "text/event-stream"},
                        stream=_ChunkedAsyncByteStream(legacy_sse, chunk_size=chunk_size),
                    )

                upstream_transport = httpx.MockTransport(_handler)
                real_async_client = httpx.AsyncClient

                def _client_factory(*args, **kwargs):
                    kwargs["transport"] = upstream_transport
                    return real_async_client(*args, **kwargs)

                asgi_transport = httpx.ASGITransport(app=gateway.app)
                async with real_async_client(transport=asgi_transport, base_url="http://gateway.local") as client:
                    with patch.object(gateway, "MODE", "alfred"), \
                         patch.object(gateway, "MASTER_KEY", "secret"), \
                         patch.object(gateway, "BACKENDS", {"local-primary": backend}), \
                         patch.object(gateway, "FALLBACKS", {}), \
                         patch.object(gateway, "TRACING_ENABLED", False), \
                         patch.object(gateway, "verify_swap_in_loaded", AsyncMock(return_value=True)), \
                         patch.object(gateway.httpx, "AsyncClient", side_effect=_client_factory):
                        resp_a = await client.post(
                            "/v1/messages",
                            headers=auth,
                            json={"model": "local-primary", "stream": True, "messages": [{"role": "user", "content": "weather?"}]},
                        )
                        resp_a2 = await client.post(
                            "/v1/messages",
                            headers=auth,
                            json={"model": "local-primary", "stream": True, "messages": [{"role": "user", "content": "weather?"}]},
                        )
                        resp_o = await client.post(
                            "/v1/chat/completions",
                            headers=auth,
                            json={"model": "local-primary", "stream": True, "messages": [{"role": "user", "content": "weather?"}]},
                        )

                self.assertEqual(200, resp_a.status_code)
                events_a = _parse_output_sse(resp_a.content)
                event_names = [et for et, _ in events_a]
                self.assertEqual("message_start", event_names[0])
                self.assertIn("content_block_start", event_names)
                self.assertIn("content_block_delta", event_names)
                self.assertIn("content_block_stop", event_names)
                self.assertEqual("message_delta", event_names[-2])
                self.assertEqual("message_stop", event_names[-1])

                start_evts = [json.loads(d) for et, d in events_a if et == "content_block_start"]
                self.assertEqual(1, len(start_evts))
                self.assertEqual(
                    {"type": "tool_use", "id": expected_id, "name": "get_weather", "input": {}},
                    start_evts[0]["content_block"],
                )
                delta_evts = [json.loads(d) for et, d in events_a if et == "content_block_delta"]
                self.assertEqual(
                    ['{"city":', '"Seattle","unit":"c"}'],
                    [e["delta"]["partial_json"] for e in delta_evts],
                )
                self.assertTrue(all(e["delta"]["type"] == "input_json_delta" for e in delta_evts))

                reconstructed = _reconstruct_anthropic_from_sse(resp_a.content)
                self.assertTrue(reconstructed["_saw_message_stop"])
                self.assertEqual("tool_use", reconstructed["stop_reason"])
                self.assertEqual({"input_tokens": 21, "output_tokens": 9}, reconstructed["usage"])
                self.assertEqual(
                    [{"type": "tool_use", "id": expected_id, "name": "get_weather", "input": {"city": "Seattle", "unit": "c"}}],
                    reconstructed["content"],
                )

                # Deterministic ID across repeated runs
                reconstructed2 = _reconstruct_anthropic_from_sse(resp_a2.content)
                self.assertEqual(reconstructed["content"], reconstructed2["content"])

                # OpenAI passthrough still forwards legacy function_call chunks + [DONE]
                self.assertEqual(200, resp_o.status_code)
                events_o = _parse_output_sse(resp_o.content)
                self.assertEqual("[DONE]", events_o[-1][1])
                objs_o = [json.loads(d) for _, d in events_o[:-1]]
                self.assertEqual("get_weather", objs_o[0]["choices"][0]["delta"]["function_call"]["name"])
                self.assertTrue(all(o["model"] == "local-primary" for o in objs_o))

    async def test_asgi_nonstream_and_buffered_legacy_function_call_translated_with_id_parity(self):
        legacy_json = {
            "id": "chatcmpl-legacy-sync",
            "object": "chat.completion",
            "created": 1700000010,
            "model": "up",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "function_call": {
                        "name": "get_weather",
                        "arguments": '{"city":"Seattle","unit":"c"}',
                    },
                },
                "finish_reason": "function_call",
            }],
            "usage": {"prompt_tokens": 21, "completion_tokens": 9, "total_tokens": 30},
        }
        expected_id = gateway._legacy_function_call_id("chatcmpl-legacy-sync", 0)
        backend = gateway.Backend("local-primary", "http://127.0.0.1:9001/v1", "up", "k")
        auth = {"Authorization": "Bearer secret"}

        def _json_handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                status_code=200,
                headers={"content-type": "application/json"},
                content=json.dumps(legacy_json).encode("utf-8"),
            )

        upstream_transport = httpx.MockTransport(_json_handler)
        real_async_client = httpx.AsyncClient

        def _client_factory(*args, **kwargs):
            kwargs["transport"] = upstream_transport
            return real_async_client(*args, **kwargs)

        asgi_transport = httpx.ASGITransport(app=gateway.app)
        async with real_async_client(transport=asgi_transport, base_url="http://gateway.local") as client:
            with patch.object(gateway, "MODE", "alfred"), \
                 patch.object(gateway, "MASTER_KEY", "secret"), \
                 patch.object(gateway, "BACKENDS", {"local-primary": backend}), \
                 patch.object(gateway, "FALLBACKS", {}), \
                 patch.object(gateway, "TRACING_ENABLED", False), \
                 patch.object(gateway, "verify_swap_in_loaded", AsyncMock(return_value=True)), \
                 patch.object(gateway.httpx, "AsyncClient", side_effect=_client_factory):
                resp_nonstream = await client.post(
                    "/v1/messages",
                    headers=auth,
                    json={"model": "local-primary", "stream": False, "messages": [{"role": "user", "content": "weather?"}]},
                )
                # Buffered JSON returned to a stream=True request goes through _json_to_fake_sse -> openai_stream_to_anthropic
                resp_buffered_stream = await client.post(
                    "/v1/messages",
                    headers=auth,
                    json={"model": "local-primary", "stream": True, "messages": [{"role": "user", "content": "weather?"}]},
                )

        self.assertEqual(200, resp_nonstream.status_code)
        body_sync = resp_nonstream.json()
        self.assertEqual("tool_use", body_sync["stop_reason"])
        self.assertEqual({"input_tokens": 21, "output_tokens": 9}, body_sync["usage"])
        self.assertEqual(
            [{"type": "tool_use", "id": expected_id, "name": "get_weather", "input": {"city": "Seattle", "unit": "c"}}],
            body_sync["content"],
        )

        self.assertEqual(200, resp_buffered_stream.status_code)
        reconstructed = _reconstruct_anthropic_from_sse(resp_buffered_stream.content)
        self.assertTrue(reconstructed["_saw_message_stop"])
        self.assertEqual(body_sync["stop_reason"], reconstructed["stop_reason"])
        self.assertEqual(body_sync["content"], reconstructed["content"])
        self.assertEqual(body_sync["usage"], reconstructed["usage"])

        # Metric helper counts legacy function_call on non-streaming responses
        self.assertEqual(1, gateway._count_tool_calls_in_chunk(legacy_json))

    async def test_asgi_text_before_legacy_function_call_zero_arg_and_modern_tool_calls_control(self):
        thinking_text_fc_sse = (
            b'data: {"id":"chatcmpl-mix","model":"up","choices":[{"index":0,"delta":{"reasoning_content":"plan first"},"finish_reason":null}]}\n\n'
            b'data: {"id":"chatcmpl-mix","model":"up","choices":[{"index":0,"delta":{"content":"Checking now."},"finish_reason":null}]}\n\n'
            b'data: {"id":"chatcmpl-mix","model":"up","choices":[{"index":0,"delta":{"function_call":{"name":"ping","arguments":""}},"finish_reason":"function_call"}],'
            b'"usage":{"prompt_tokens":10,"completion_tokens":4}}\n\n'
            b"data: [DONE]\n\n"
        )
        modern_tool_sse = (
            b'data: {"id":"chatcmpl-modern","model":"up","choices":[{"index":0,"delta":{"role":"assistant",'
            b'"tool_calls":[{"index":0,"id":"call_modern_1","type":"function","function":{"name":"lookup","arguments":""}}]},"finish_reason":null}]}\n\n'
            b'data: {"id":"chatcmpl-modern","model":"up","choices":[{"index":0,"delta":{'
            b'"tool_calls":[{"index":0,"function":{"arguments":"{\\"ticker\\":\\"NVDA\\"}"}}]},"finish_reason":"tool_calls"}],'
            b'"usage":{"prompt_tokens":12,"completion_tokens":6}}\n\n'
            b"data: [DONE]\n\n"
        )
        modern_tool_json = {
            "id": "chatcmpl-modern-sync",
            "object": "chat.completion",
            "created": 1700000011,
            "model": "up",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_modern_sync",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": '{"ticker":"NVDA"}'},
                    }],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 12, "completion_tokens": 6, "total_tokens": 18},
        }
        backend = gateway.Backend("local-primary", "http://127.0.0.1:9001/v1", "up", "k")
        auth = {"Authorization": "Bearer secret"}

        cases = [
            ("thinking_text_zero_arg_legacy_fc", True, thinking_text_fc_sse, "text/event-stream"),
            ("modern_tool_calls_stream_control", True, modern_tool_sse, "text/event-stream"),
            ("modern_tool_calls_sync_control", False, json.dumps(modern_tool_json).encode("utf-8"), "application/json"),
        ]
        for label, is_stream, payload, ctype in cases:
            with self.subTest(case=label):
                def _handler(_request: httpx.Request, _p=payload, _ct=ctype, _s=is_stream) -> httpx.Response:
                    if _s:
                        return httpx.Response(
                            status_code=200,
                            headers={"content-type": _ct},
                            stream=_ChunkedAsyncByteStream(_p, chunk_size=17),
                        )
                    return httpx.Response(status_code=200, headers={"content-type": _ct}, content=_p)

                upstream_transport = httpx.MockTransport(_handler)
                real_async_client = httpx.AsyncClient

                def _client_factory(*args, **kwargs):
                    kwargs["transport"] = upstream_transport
                    return real_async_client(*args, **kwargs)

                asgi_transport = httpx.ASGITransport(app=gateway.app)
                async with real_async_client(transport=asgi_transport, base_url="http://gateway.local") as client:
                    with patch.object(gateway, "MODE", "alfred"), \
                         patch.object(gateway, "MASTER_KEY", "secret"), \
                         patch.object(gateway, "BACKENDS", {"local-primary": backend}), \
                         patch.object(gateway, "FALLBACKS", {}), \
                         patch.object(gateway, "TRACING_ENABLED", False), \
                         patch.object(gateway, "verify_swap_in_loaded", AsyncMock(return_value=True)), \
                         patch.object(gateway.httpx, "AsyncClient", side_effect=_client_factory):
                        resp = await client.post(
                            "/v1/messages",
                            headers=auth,
                            json={"model": "local-primary", "stream": is_stream, "messages": [{"role": "user", "content": "run"}]},
                        )

                self.assertEqual(200, resp.status_code)
                if label == "thinking_text_zero_arg_legacy_fc":
                    rec = _reconstruct_anthropic_from_sse(resp.content)
                    self.assertTrue(rec["_saw_message_stop"])
                    self.assertEqual("tool_use", rec["stop_reason"])
                    self.assertEqual(
                        [
                            {"type": "thinking", "thinking": "plan first"},
                            {"type": "text", "text": "Checking now."},
                            {
                                "type": "tool_use",
                                "id": gateway._legacy_function_call_id("chatcmpl-mix", 0),
                                "name": "ping",
                                "input": {},
                            },
                        ],
                        rec["content"],
                    )
                elif label == "modern_tool_calls_stream_control":
                    rec = _reconstruct_anthropic_from_sse(resp.content)
                    self.assertTrue(rec["_saw_message_stop"])
                    self.assertEqual("tool_use", rec["stop_reason"])
                    self.assertEqual(
                        [{"type": "tool_use", "id": "call_modern_1", "name": "lookup", "input": {"ticker": "NVDA"}}],
                        rec["content"],
                    )
                else:
                    body = resp.json()
                    self.assertEqual("tool_use", body["stop_reason"])
                    self.assertEqual(
                        [{"type": "tool_use", "id": "call_modern_sync", "name": "lookup", "input": {"ticker": "NVDA"}}],
                        body["content"],
                    )


if __name__ == "__main__":
    unittest.main()

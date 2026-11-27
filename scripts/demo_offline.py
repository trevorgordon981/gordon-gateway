#!/usr/bin/env python3
"""5-minute (actually ~1 second) offline mock-upstream demo for gordon-gateway.

Starts real loopback mock LLM HTTP servers on ephemeral 127.0.0.1 ports and
drives gordon-gateway end-to-end with zero external network access, zero API
keys, and zero GPU requirements.

Scenarios verified live:
  1. Fragmented CRLF SSE -> Anthropic /v1/messages translation (thinking + text + tool_use)
  2. Premature EOF detection -> emits Anthropic `event: error`, never `message_stop`
  3. Failover & recovery -> primary 503 automatically recovers on local backup backend
  4. Terminal 400 & data-boundary guard -> 400 never retries; cloud fallback blocked by default
  5. Walter mode -> `tool_choice="none"` skips tool prompt injection; `auto` parses TOOL_CALL
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Configure isolated offline demo environment before importing gateway.py
_TMP = tempfile.TemporaryDirectory(prefix="gordon-gateway-demo-")
os.environ["GATEWAY_MODE"] = "alfred"
os.environ["MASTER_KEY"] = "demo-master-key"
os.environ["TRACING_ENABLED"] = "false"
os.environ["GATEWAY_CONFIG"] = str(Path(_TMP.name) / "no-config.yaml")
os.environ["LOG_FILE"] = str(Path(_TMP.name) / "gateway.log")
os.environ["BLOCKOPS_LOG_FILE"] = str(Path(_TMP.name) / "blockops.log")

import gateway  # noqa: E402


class MockUpstreamServer:
    """Minimal asyncio HTTP/1.1 mock upstream supporting JSON, errors, and fragmented CRLF SSE."""

    def __init__(self, handler):
        self._handler = handler
        self._server: asyncio.AbstractServer | None = None
        self.port: int = 0
        self.requests: list[dict] = []

    async def start(self) -> str:
        self._server = await asyncio.start_server(self._handle_conn, "127.0.0.1", 0)
        self.port = self._server.sockets[0].getsockname()[1]
        return f"http://127.0.0.1:{self.port}/v1"

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

    async def _handle_conn(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            raw_head = await reader.readuntil(b"\r\n\r\n")
            lines = raw_head.decode("latin1").split("\r\n")
            method, path, _ = lines[0].split(" ", 2)
            headers: dict[str, str] = {}
            for line in lines[1:]:
                if ":" in line:
                    k, v = line.split(":", 1)
                    headers[k.strip().lower()] = v.strip()
            cl = int(headers.get("content-length", "0"))
            raw_body = await reader.readexactly(cl) if cl > 0 else b""
            payload = json.loads(raw_body.decode("utf-8")) if raw_body else {}
            self.requests.append({"method": method, "path": path, "body": payload})
            status, ctype, chunks = self._handler(path, payload)
            head = (
                f"HTTP/1.1 {status} OK\r\n"
                f"Content-Type: {ctype}\r\n"
                f"Connection: close\r\n"
            )
            if len(chunks) == 1 and "event-stream" not in ctype:
                head += f"Content-Length: {len(chunks[0])}\r\n"
            head += "\r\n"
            writer.write(head.encode("ascii"))
            await writer.drain()
            for chunk in chunks:
                writer.write(chunk)
                await writer.drain()
                await asyncio.sleep(0.002)
        except Exception:
            pass
        finally:
            writer.close()
            await writer.wait_closed()


def _sse_frame(obj: dict | str, sep: bytes = b"\r\n") -> bytes:
    payload = obj.encode("utf-8") if isinstance(obj, str) else json.dumps(obj).encode("utf-8")
    return b"data: " + payload + sep + sep


async def run_demo() -> None:
    print("== gordon-gateway offline mock-upstream demo ==\n")

    # 1. Mock primary (configurable behavior per prompt)
    def primary_handler(_path: str, body: dict):
        prompt = str(body.get("messages", [{}])[-1].get("content", ""))
        if "TRIGGER_503" in prompt:
            return 503, "application/json", [b'{"error":{"message":"primary overloaded"}}']
        if "TRIGGER_400" in prompt:
            return 400, "application/json", [b'{"error":{"message":"invalid parameter"}}']
        if "PREMATURE_EOF" in prompt:
            partial = _sse_frame({
                "id": "demo-eof",
                "model": "mock-primary",
                "choices": [{"index": 0, "delta": {"content": "cut off mid-sentence"}, "finish_reason": None}],
            })
            return 200, "text/event-stream", [partial]
        # Default: fragmented CRLF SSE stream with reasoning + text + tool call
        full = b"".join([
            b": keepalive\r\n\r\n",
            _sse_frame({
                "id": "demo-1",
                "model": "mock-primary",
                "choices": [{"index": 0, "delta": {"role": "assistant", "reasoning_content": "plan tool call"}, "finish_reason": None}],
            }),
            _sse_frame({
                "id": "demo-1",
                "model": "mock-primary",
                "choices": [{"index": 0, "delta": {"content": "Checking weather..."}, "finish_reason": None}],
            }),
            _sse_frame({
                "id": "demo-1",
                "model": "mock-primary",
                "choices": [{
                    "index": 0,
                    "delta": {"tool_calls": [{"index": 0, "id": "call_w1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city":"NYC"}'}}]},
                    "finish_reason": "tool_calls",
                }],
                "usage": {"prompt_tokens": 20, "completion_tokens": 11},
            }),
            _sse_frame("[DONE]"),
        ])
        # Fragment into tiny 7-byte chunks so CRLF delimiters split across chunk boundaries
        return 200, "text/event-stream", [full[i:i + 7] for i in range(0, len(full), 7)]

    # 2. Mock local backup
    def backup_handler(_path: str, body: dict):
        resp = {
            "id": "demo-backup",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "mock-backup",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Recovered cleanly on local backup!"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 14, "completion_tokens": 7, "total_tokens": 21},
        }
        return 200, "application/json", [json.dumps(resp).encode("utf-8")]

    # 3. Mock walter backend (returns prompt-injected TOOL_CALL syntax)
    def walter_handler(_path: str, body: dict):
        text = "Sure, checking now.\nTOOL_CALL: {\"name\": \"ping\", \"arguments\": {\"host\": \"localhost\"}}"
        if not any("Available functions:" in str(m.get("content", "")) for m in body.get("messages", [])):
            text = "Direct reply without tools."
        resp = {
            "id": "demo-walter",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "qwen3.5-122b-a10b-8bit",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 18, "completion_tokens": 9, "total_tokens": 27},
        }
        return 200, "application/json", [json.dumps(resp).encode("utf-8")]

    primary_srv = MockUpstreamServer(primary_handler)
    backup_srv = MockUpstreamServer(backup_handler)
    walter_srv = MockUpstreamServer(walter_handler)

    primary_url = await primary_srv.start()
    backup_url = await backup_srv.start()
    walter_url = await walter_srv.start()
    walter_base = walter_url[:-3]  # strip /v1

    try:
        backends = {
            "demo-model": gateway.Backend("demo-model", primary_url, "mock-primary", "not-needed"),
            "demo-backup": gateway.Backend("demo-backup", backup_url, "mock-backup", "not-needed"),
            "cloud-fallback": gateway.Backend(
                "cloud-fallback",
                "https://openrouter.ai/api/v1",
                "remote-model",
                "not-needed",
                trust_boundary="external",
            ),
        }
        fallbacks = {"demo-model": ["demo-backup", "cloud-fallback"]}

        transport = httpx.ASGITransport(app=gateway.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://gateway.local") as client:
            auth = {"Authorization": "Bearer demo-master-key"}

            with patch.object(gateway, "MODE", "alfred"), \
                 patch.object(gateway, "BACKENDS", backends), \
                 patch.object(gateway, "FALLBACKS", fallbacks), \
                 patch.object(gateway, "ALLOW_EXTERNAL_FALLBACK", False), \
                 patch.object(gateway, "RETRYABLE_4XX_STATUSES", frozenset()):

                # Scenario 1: Fragmented CRLF SSE -> Anthropic /v1/messages
                r1 = await client.post(
                    "/v1/messages",
                    headers=auth,
                    json={"model": "demo-model", "stream": True, "messages": [{"role": "user", "content": "weather in NYC?"}]},
                )
                assert r1.status_code == 200, r1.text
                events1, _ = gateway.parse_sse_blocks(r1.content, final=True)
                etypes1 = [et for et, _ in events1]
                assert "message_stop" in etypes1 and "error" not in etypes1
                assert b"plan tool call" in r1.content and b"Checking weather..." in r1.content and b"get_weather" in r1.content
                print("[1/5] PASS  Fragmented CRLF SSE -> Anthropic thinking + text + tool_use preserved")

                # Scenario 2: Premature EOF -> error event, NEVER message_stop
                r2 = await client.post(
                    "/v1/messages",
                    headers=auth,
                    json={"model": "demo-model", "stream": True, "messages": [{"role": "user", "content": "PREMATURE_EOF"}]},
                )
                events2, _ = gateway.parse_sse_blocks(r2.content, final=True)
                etypes2 = [et for et, _ in events2]
                assert etypes2[-1] == "error" and "message_stop" not in etypes2
                print("[2/5] PASS  Premature EOF stream emitted `event: error` (no false `message_stop`)")

                # Scenario 3: 503 failover & recovery to local backup
                backup_before = len(backup_srv.requests)
                r3 = await client.post(
                    "/v1/chat/completions",
                    headers=auth,
                    json={"model": "demo-model", "messages": [{"role": "user", "content": "TRIGGER_503"}]},
                )
                assert r3.status_code == 200
                assert len(backup_srv.requests) == backup_before + 1
                assert "Recovered cleanly" in r3.json()["choices"][0]["message"]["content"]
                print("[3/5] PASS  HTTP 503 failed over from primary -> local backup and recovered (HTTP 200)")

                # Scenario 4: Terminal HTTP 400 never retries + external cloud blocked by default
                backup_before = len(backup_srv.requests)
                r4 = await client.post(
                    "/v1/chat/completions",
                    headers=auth,
                    json={"model": "demo-model", "messages": [{"role": "user", "content": "TRIGGER_400"}]},
                )
                assert r4.status_code == 400
                assert len(backup_srv.requests) == backup_before, "400 must not advance fallback chain"
                resolved_names = [b.name for b in gateway.resolve_chain("demo-model")]
                assert resolved_names == ["demo-model", "demo-backup"], f"cloud must be blocked by default: {resolved_names}"
                print("[4/5] PASS  HTTP 400 terminated immediately (0 backup calls) and cloud fallback blocked by default")

            # Scenario 5: Walter mode tool_choice='none' vs tool_choice='auto'
            gateway._session_states.clear()
            gateway._inflight_per_backend.clear()
            with patch.object(gateway, "MODE", "walter"), \
                 patch.object(gateway, "WALTER_BACKEND", walter_base), \
                 patch.object(gateway, "MODEL_BACKENDS", {"qwen35": walter_base, gateway.WALTER_CANONICAL_MODEL: walter_base}):
                tools = [{"type": "function", "function": {"name": "ping", "description": "ping host", "parameters": {"type": "object", "properties": {"host": {"type": "string"}}}}}]
                r5a = await client.post(
                    "/v1/chat/completions",
                    headers={"X-Gordon-Session-Id": "demo-walter-none"},
                    json={"model": "qwen35", "tools": tools, "tool_choice": "none", "messages": [{"role": "user", "content": "hi"}]},
                )
                assert r5a.status_code == 200
                assert "Direct reply without tools." in r5a.json()["choices"][0]["message"]["content"]

                r5b = await client.post(
                    "/v1/chat/completions",
                    headers={"X-Gordon-Session-Id": "demo-walter-auto"},
                    json={"model": "qwen35", "tools": tools, "tool_choice": "auto", "messages": [{"role": "user", "content": "ping localhost"}]},
                )
                assert r5b.status_code == 200
                assert b'"name": "ping"' in r5b.content and b'"finish_reason": "tool_calls"' in r5b.content
                print("[5/5] PASS  Walter mode honored `tool_choice=none` and parsed `TOOL_CALL:` on `auto`")

        print("\nAll 5 offline demo scenarios passed.")
    finally:
        await primary_srv.stop()
        await backup_srv.stop()
        await walter_srv.stop()
        _TMP.cleanup()


if __name__ == "__main__":
    asyncio.run(run_demo())

# Changelog

All notable changes to `gordon-gateway` are documented in this file.

## [0.1.0] - 2026-11-27

### Added

- **OpenAI and Anthropic protocol translation**: Unified `/v1/chat/completions`, `/v1/messages`, and `/v1/messages/count_tokens` handling in `alfred` mode, translating `reasoning_content` / `reasoning` to Anthropic `thinking` blocks, `tool_choice` modes, multimodal images (`base64` and `url`), and legacy OpenAI `function_call` payloads to Anthropic `tool_use` blocks.
- **Incremental LF/CRLF/CR SSE framing and completion guard**: Stream parser and translators (`parse_sse_blocks()`, `_has_openai_completion_evidence()`, `_open_upstream_sse_stream()`, `rewrite_stream_model()`, `openai_stream_to_anthropic()`) preserve fragmented CRLF frames, require real completion evidence before committing HTTP `200` or emitting terminal success frames, emit explicit `event: error` on empty, usage-only, or prematurely closed streams, and enforce a consistent post-`[DONE]` terminal-marker policy.
- **Fail-closed retry and trust-boundary failover**: Retries bounded `httpx.TransportError`, HTTP `5xx`, malformed/non-JSON `200`, and empty pre-commit SSE failures across configured local fallback chains while keeping HTTP `4xx` terminal by default and blocking local-to-external cloud fallback unless explicitly authorized with typed booleans or allowlisted routes.
- **Pooled upstream HTTP client with FastAPI lifespan**: Managed shared `httpx.AsyncClient` connection pool via FastAPI ASGI `lifespan` with clean shutdown teardown.
- **Walter mode tool adaptation and session guards**: Prompt-injected `TOOL_CALL:` parser (including recovered span stripping), explicit `tool_choice` enforcement (`auto`, `none`, and typed HTTP `400` on unsupported forced choices), and opaque SHA-256 per-session LRU+TTL loop, cycle, context, and memory guards.
- **Privacy-first telemetry and secrets contract**: Metadata-only OTLP export by default (`TRACE_CONTENT=false`), authenticated TLS requirement for non-loopback OTLP endpoints, and owner-only (`0600`) secret-file loading.
- **Offline verification suite and CI**: End-to-end offline mock-upstream demo (`scripts/demo_offline.py`), 58-test / 118-subtest offline suite (`tests/` and `test_gateway_sessions.py`), and GitHub Actions CI workflow.

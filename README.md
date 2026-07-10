# gordon-gateway

A single, config-driven LLM gateway proxy that unifies **three** previously
separate proxies into one codebase with a modular middleware chain. Same code,
two modes:

| Mode | Runs on | Replaces | Default port |
|------|---------|----------|--------------|
| `alfred` | bat-studio | `alfred-llm-proxy` (routing/translation) **+** `metrics-proxy` (OTLP tracing) | **8091** (Studio override: **8095**) |
| `walter` | MacBook | `blockops-proxy` (tool-call translation/loop guards) **+** OTLP tracing layered on | **8080** |

> **Deployment status:** Gordon Gateway is active on Studio as
> `ai.alfred.llm-proxy`, bound to `127.0.0.1:8095` in `alfred` mode. Studio's
> inference backend is Trevor's custom Python serving engine—not vMLX or vLLM.
> The running Byron benchmark calls that engine directly on `:8082` and does
> not traverse this gateway.

Framework: **FastAPI + uvicorn + httpx** (the studio pyenv `3.12.13` already
has these plus `opentelemetry` and `psutil`). The blockops parsing / loop-guard
/ truncation logic was ported verbatim from the local `blockops-proxy.py`; its
`aiohttp` HTTP layer was re-implemented on `httpx` so the whole app runs on one
framework with **no new dependencies**.

## Middleware chain

Every chat request flows through the same conceptual pipeline; which stages do
real work depends on `mode`:

```
  route (model selection)
    -> translate-request   (alfred: Anthropic->OpenAI ; walter: tool-prompt inject + sanitize)
      -> trace-start       (OTLP "chat" span: gen_ai.* + hermes.user_* attrs, prompt event)
        -> forward         (httpx -> backend; alfred = fallback chain, walter = single backend)
      -> translate-response (alfred: OpenAI->Anthropic / model-id rewrite ;
                             walter: TOOL_CALL: text -> tool_calls, strip <think>, loop/ctx/mem guards)
    -> trace-end           (span end: usage tokens, tool_call count, latency; record metrics + Sonnet-$ )
```

- **route** — `alfred`: `resolve_chain()` maps the requested model id to a
  `Backend` + ordered fallback list. `walter`: `_backend_for_model()` +
  `MODEL_ALIASES`.
- **translate-request** — `alfred`: `anthropic_to_openai_request()` on
  `/v1/messages`; `/v1/chat/completions` passes through. `walter`:
  `build_tool_system_prompt()` + `inject_tool_prompt()` +
  `sanitize_messages_for_backend()`; loop-guard nudges; context truncation.
- **trace** — single OTLP layer (`_start_chat_span` / `_record_metrics`) used by
  **both** modes. Emits `chat` spans + the same 7 metrics + Sonnet-equivalent
  cost as the old metrics-proxy, with persistent counter state restore.
- **forward** — `alfred`: streaming via `_stream_upstream_chat` (handles
  backend that answers `stream:true` with buffered JSON), non-stream via httpx;
  fallback chain on connect/5xx. `walter`: non-stream tool path with SSE
  heartbeats; streaming/non-stream no-tool path.
- **translate-response** — `alfred`: `rewrite_stream_model` /
  `openai_stream_to_anthropic` / `openai_to_anthropic_response`. `walter`:
  `parse_tool_calls` (5 formats + recovery), `rewrite_response`, `strip_thinking`.

### Resilient fallback-to-direct-backend
- `alfred`: `FALLBACKS` chain (e.g. `qwen-397b -> openrouter-free`); on
  identity-check fail / connect error / upstream 5xx it advances to the next
  backend, returning 502 only when all fail.
- `walter`: backend connect/parse failures return a clean 502 JSON instead of
  hanging; `_walter_no_tools_path` is itself the "direct backend" route.
- `/health` (always 200 + mode/tracing) and `/health/deep` (active per-backend
  probe) on both modes.

## Feature-parity matrix

### alfred-llm-proxy -> gateway (alfred mode)

| Old feature (proxy.py) | Where it lives now |
|---|---|
| Listen `127.0.0.1:8091` (PORT env) | `MODE=alfred` default port 8091; `PORT`/`port` override |
| `MASTER_KEY` bearer auth (`hmac.compare_digest`) | `check_auth()`; required in alfred mode |
| Backend table (qwen-397b, kimi, glm-5.1, glm-5.1-test, qwen-alfred-router, openrouter-free) | `_default_alfred_backends()` (verbatim) / `backends:` config |
| Fallback chains | `FALLBACKS` / `fallbacks:` config |
| Swap-in identity check (`/v1/models` data[0].id, 5s cache) | `verify_swap_in_loaded()` (verbatim) |
| Header blocklist / forwarding | `HEADER_BLOCKLIST`, `forwarded_headers()`, `make_upstream_headers()` |
| 50MB body cap | `MAX_BODY_BYTES` |
| `/v1/models`, `/health*`, `/health/deep` | same routes |
| `/v1/chat/completions` (stream + non-stream, model-id rewrite) | `chat_completions()` + `rewrite_stream_model()` |
| `/v1/messages` Anthropic translation (incl. reasoning_content -> thinking blocks, tool-call assembly) | `anthropic_to_openai_request`, `openai_to_anthropic_response`, `openai_stream_to_anthropic` (verbatim) |
| `/v1/messages/count_tokens` heuristic | `count_tokens()` |
| buffered-JSON-on-stream -> fake SSE | `_json_to_fake_sse()` |
| Rotating file log | `LOG_CONFIG` / `gateway.log` |

### metrics-proxy -> gateway (trace/metrics layer, BOTH modes)

| Old feature (metrics_proxy.py) | Where it lives now |
|---|---|
| OTLP `chat` spans (`gen_ai.*`, `hermes.user_id/class`) | `_start_chat_span()` + per-mode span finalize |
| 7 metrics: tokens in/out/total, tool_calls, requests, duration histogram, Sonnet-equiv cost | `_init_otel()` / `_record_metrics()` |
| `X-Hermes-User` attribution (primary/other/unknown) | `_resolve_user()` (verbatim) |
| Persistent counter state + restore on boot, periodic save | `_load_state` / `_save_state` / `_restore_counters` |
| `stream_options.include_usage` injection so streaming usage is captured | ON in both modes: alfred `/v1/messages` (`anthropic_to_openai_request`) + alfred `/v1/chat/completions` streaming + walter no-tools streaming path (see GAP-3 — resolved) |
| streaming span ended in generator `finally` with usage/tool/latency | `_alfred_traced_stream()` (alfred) / generator `finally` in `_walter_tools_path` |
| `TRACE_CONTENT` prompt/completion span events | `TRACE_CONTENT`; prompt event in `_start_chat_span` |
| Sonnet pricing env-overridable | `SONNET_INPUT_PRICE` / `SONNET_OUTPUT_PRICE` |
| `/health` | `/health` (returns mode+tracing) |

### blockops-proxy -> gateway (walter mode)

| Old feature (blockops-proxy.py, local) | Where it lives now |
|---|---|
| Listen `:8080`, `BACKEND_URL` default `:8082` | `MODE=walter` default 8080; `walter_backend`/`BACKEND_URL` |
| `BACKEND_STRIP_PATH_PREFIX` path rewrite | `_rewrite_backend_path()` |
| 5-format tool-call parsing + JSON recovery (bare-name, missing-args, brace-append, bare-JSON) | `parse_tool_calls()` (verbatim) |
| `<think>` / "Thinking Process:" stripping | `strip_thinking()` (verbatim) |
| Tool system prompt build + padding-to-stable-prefix + cache | `build_tool_system_prompt()` |
| Inject tool prompt into LAST user msg (cache-stable) | `inject_tool_prompt()` |
| `sanitize_messages_for_backend` (tool_calls/tool-role flatten + result compression) | same (verbatim) |
| Loop guards: identical thresholds, isolated per conversation/agent | `detect_tool_loop`, `detect_tool_cycle`, `WalterSessionState` |
| Repeat-parse-error hard stop | Per-session `parse_error_counts` + REPEAT_PARSE_ERROR_STOP path |
| Context guards (warn 80k / hard 100k truncate) | `truncate_messages()` + thresholds |
| Memory-pressure warnings (psutil tiers) | `get_memory_warning()`, `MEM_WARN_TIERS` |
| Session-start banner + testing-mode notice | Per-session `banner_sent`, `make_session_start_notice()` |
| `/new`/message-count reset | Clears only that session's state; shared backend cache is retained |
| Shared cache janitor | Global purge is allowed only after all backends are idle and the size threshold is exceeded |
| Per-backend concurrency cap (429 w/ Retry-After) | `_inc/_dec/_get_inflight`, MAX_CONCURRENT |
| SSE heartbeats during inference; tool path streams tool_calls | `_walter_tools_path()` generator |
| Synthetic `/v1/models/{id}`, passthrough for non-chat | `walter_passthrough()` |
| Append-only traffic log w/ batched flush | `bolog()` (separate `BLOCKOPS_LOG_FILE`) |
| **Default alias map** (`MODEL_ALIASES`, studio's ~8: `qwen35`, `coder`, `reason`, `r1`, …) | `_default_walter_aliases()` / `model_aliases:` config |
| **Default per-model backend map** (`MODEL_BACKENDS`) | `_default_walter_backends()` / `model_backends:` config |
| **`normalize_upstream_model()`** (alias OR unknown id -> canonical path; avoids cold model swap) | `normalize_upstream_model()` (verbatim); called in BOTH walter paths |
| **`CODER_BACKEND` (:8083) / `REASON_BACKEND` (:8084)** per-role routing | `CODER_BACKEND` / `REASON_BACKEND` consts -> `_backend_for_model()` |

### Walter session isolation

Conversation state is keyed in this order: `X-Claude-Code-Session-Id`,
`X-Gordon-Session-Id`, `X-Hermes-Session-Id`, `X-Session-Id`,
`X-Conversation-Id`, `metadata.session_id`/`conversation_id`/`thread_id`, then a
deterministic hash of the user plus the initial system/user messages. Claude
Code agent IDs are included so concurrent subagents do not share loop counters.
Raw identifiers are hashed before storage or logging.

The state table is a TTL-bound LRU (`session_state_max`,
`session_state_ttl_seconds`), and malformed-call signatures are separately
bounded per session. Clients other than Claude Code should send
`X-Gordon-Session-Id` for unambiguous isolation; the root-message fallback
cannot distinguish two sessions that deliberately begin with identical input.
A shorter message list clears only that entry's banner/counters/accounting. It
never purges the backend's shared cache.

## Parity GAPS / risks found

- **GAP-1 (RESOLVED) — walter mode covers BOTH blockops forks, with one
  intentional safety change.** The studio's live `~alfredpennyworth/blockops-proxy.py` is a
  more-developed fork than the local/public `~/code/blockops-proxy/blockops-proxy.py`.
  Walter mode folds in every studio-only feature while preserving laptop
  pass-through behaviour by default-on-empty-map. A conversation reset no
  longer purges a cache shared by unrelated sessions; the idle, no-inflight
  janitor remains. Proof matrix:

  | Feature | Public/laptop blockops | Studio blockops | gateway walter |
  |---|---|---|---|
  | 5-format tool-call parse + recovery | yes | yes | yes (`parse_tool_calls`) |
  | `<think>` / "Thinking Process:" strip | yes | yes | yes (`strip_thinking`) |
  | loop / cycle / session guards | yes | yes | yes (per-session isolated) |
  | context truncate + memory tiers | yes | yes | yes |
  | reset purge + idle janitor | yes | yes | reset purge intentionally removed; safe idle janitor retained |
  | per-backend concurrency cap | yes | yes | yes |
  | `_backend_for_model()` routing | yes | yes | yes |
  | **`MODEL_ALIASES` populated (~8)** | no (empty) | yes | yes — `_default_walter_aliases()` |
  | **`MODEL_BACKENDS` populated** | no (empty) | yes | yes — `_default_walter_backends()` |
  | **`normalize_upstream_model()` coercion** | no | yes | yes — called in both walter paths |
  | **`CODER_BACKEND` / `REASON_BACKEND`** | no | defined (unused) | yes — **wired** via `_backend_for_model` |

  - **Parity preserved both ways.** The defaults reproduce the studio. Set
    `model_aliases: {}` (explicit empty) in `config.yaml` to get the *public*
    pass-through behaviour exactly: `normalize_upstream_model()` is a no-op when
    `MODEL_ALIASES` is empty, and `_backend_for_model()` then returns
    `WALTER_BACKEND` for everything. So one codebase covers both forks; which one
    you get is a config choice.
  - **Superset note (intentional improvement, not a skip):** the studio file
    *defines* `CODER_BACKEND`/`REASON_BACKEND` but never wires them
    (its `MODEL_BACKENDS` sends every alias to `DEFAULT_BACKEND`). The gateway
    actually routes `coder`/`qwen-coder` -> `CODER_BACKEND` and
    `reason`/`qwen-reason`/`r1` -> `REASON_BACKEND` in `_default_walter_backends()`.
    Both default to `walter_backend`, so on the single-server laptop this is a
    no-op; on a multi-server host it does what the studio constants always
    implied. Override per-role with `coder_backend:` / `reason_backend:` or the
    full `model_backends:` map.
  - **Intentionally NOT ported from the studio fork:** studio-only experimental
    knobs unrelated to alias/backend routing — `TOOL_PARSE_MAX_RETRIES` /
    `TOOL_RETRY_NUDGE_TEMPLATE` (truncated-tool-call auto-retry) and the
    empty-`<think>` retry path. These are orthogonal to the two enhancements in
    scope (alias/backend feature-completeness + o11y usage injection), are not
    present in the public fork, and touch the tool-call retry loop; folding them
    in is a separate, larger change. Flagged here as the only remaining studio
    delta. Everything alias/backend/normalize-related is in.
- **GAP-2 (CONFIGURATION-SENSITIVE) — backend targets must describe the custom
  Python serving topology.** The active gateway listens on `:8095`; Byron talks
  directly to the custom engine on `:8082`. Never point a gateway backend back
  at its own listen port, and do not infer a vMLX/vLLM activation path from
  legacy model names. Backend URLs remain deployment-local configuration.
- **GAP-3 (RESOLVED) — usage-injection ON, both modes, stream + non-stream.**
  Streaming requests now always capture complete token usage in the trace layer.
  Where injection happens:

  | Mode / endpoint | Stream | Non-stream |
  |---|---|---|
  | alfred `/v1/chat/completions` | injects `stream_options.include_usage=true` into `upstream_body` (after `upstream_body["model"]=…`) | usage read from response body (no injection needed) |
  | alfred `/v1/messages` | injects it in `anthropic_to_openai_request()` (unchanged) | usage from response |
  | walter no-tools path | injects it on the upstream body before send (guarded by `client_wants_stream`) | usage from response / estimate fallback |
  | walter tools path | upstream is forced non-stream (`data["stream"]=False`); usage read from buffered response | n/a |

  Result: a streaming client that does NOT itself request usage no longer traces
  `in=0/out=0` — the gateway adds the flag so the upstream emits a final usage
  chunk. Non-stream paths already had usage from the response body. The OTLP
  trace layer is **active in both modes** (`_init_otel()` runs unconditionally at
  startup, gated by `TRACING_ENABLED`; `_start_chat_span` + `_record_metrics`
  fire in alfred via `_alfred_traced_stream`, and in walter via `_walter_trace`).
  Injection only ever ADDS `include_usage` and merges with any client-supplied
  `stream_options`, so it never clobbers a client that set its own.
- **GAP-4 (LOW) — walter tracing is new.** The local blockops had NO tracing;
  walter mode adds it (per the unification spec). It is on by default
  (`TRACING_ENABLED=true`) but **degrades to no-op** if the otel libs are
  missing or `OTEL_ENDPOINT` is unreachable (BatchSpanProcessor drops silently),
  so it cannot break serving. Set `TRACING_ENABLED=false` on the laptop if no
  collector is reachable.
- **GAP-5 (LOW) — walter heartbeat cadence.** The original wrote a heartbeat,
  then waited up to HEARTBEAT_INTERVAL on the backend task. The port keeps the
  same loop but yields heartbeats through the StreamingResponse generator; cadence
  is equivalent but exact byte timing may differ by sub-second. No client impact.
- **GAP-6 (LOW) — `@app.on_event` deprecation.** Uses FastAPI's
  `on_event("startup"/"shutdown")` to match the metrics-proxy original exactly.
  Works on the installed FastAPI; migrate to lifespan handlers later if desired.
- **Note — walter catch-all route** is registered at startup (after the explicit
  routes) so `/v1/chat/completions` etc. win; the catch-all only handles
  health/models/cache/other passthrough.

## Studio deployment and deferred activation

- Active process: `ai.alfred.llm-proxy`, `GATEWAY_MODE=alfred`,
  `127.0.0.1:8095`.
- Session isolation is active and verified with deterministic unit and isolated
  integration tests.
- `gateway.py` loads only `MASTER_KEY` and `OPENROUTER_API_KEY` from the
  owner-only file named by `GATEWAY_SECRET_ENV` (default `~/.hermes/.env`).
  Credentials must never be embedded in a launchd plist, Git, documentation,
  or a command argument.
- The secret-file/plist migration is activated only after the in-flight Byron
  benchmark finishes. It requires restarting the gateway and its authenticated
  clients, but never the custom Python serving engine.

Verification after activation:

1. Confirm the secret file and encrypted SOPS copy are both mode `0600` and
   round-trip to identical key/value maps.
2. Confirm the launchd plist contains no credential values.
3. Restart only `ai.alfred.llm-proxy`; `GET http://127.0.0.1:8095/health` must
   report `mode=alfred` and authenticated requests must succeed with the new
   key while the previous key returns `401`.
4. Exercise two distinct session IDs in an isolated test. Do not send a live
   model request while Byron is running.

Rollback is file-level: restore the previous reviewed `gateway.py` and plist,
then restart only `ai.alfred.llm-proxy`. Do not restart, signal, or purge the
custom Python serving engine.

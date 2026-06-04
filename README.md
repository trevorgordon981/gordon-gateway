# gordon-gateway

A single, config-driven LLM gateway proxy that unifies **three** previously
separate proxies into one codebase with a modular middleware chain. Same code,
two modes:

| Mode | Runs on | Replaces | Default port |
|------|---------|----------|--------------|
| `alfred` | bat-studio | `alfred-llm-proxy` (routing/translation) **+** `metrics-proxy` (OTLP tracing) | **8091** |
| `walter` | MacBook | `blockops-proxy` (tool-call translation/loop guards) **+** OTLP tracing layered on | **8080** |

> **STAGING ONLY.** This directory is built but NOT deployed. The Alfred stack
> is intentionally down for a multi-day eval; cutover happens post-eval (see
> checklist below). Nothing here touches a live service.

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
| Loop guards: identical (nudge 4 / stop 8), cycle (nudge 3 / stop 4 reps), session nudge limit 8 | `detect_tool_loop`, `detect_tool_cycle`, `_assistant_tool_signature` (verbatim) |
| Repeat-parse-error hard stop | `_parse_error_counts` + REPEAT_PARSE_ERROR_STOP path |
| Context guards (warn 80k / hard 100k truncate) | `truncate_messages()` + thresholds |
| Memory-pressure warnings (psutil tiers) | `get_memory_warning()`, `MEM_WARN_TIERS` |
| Session-start banner + testing-mode notice | `make_session_start_notice()`, banner emit |
| KV-cache purge on `/new` + idle cache janitor | `purge_vmlx_cache()`, `cache_janitor()` |
| Per-backend concurrency cap (429 w/ Retry-After) | `_inc/_dec/_get_inflight`, MAX_CONCURRENT |
| SSE heartbeats during inference; tool path streams tool_calls | `_walter_tools_path()` generator |
| Synthetic `/v1/models/{id}`, passthrough for non-chat | `walter_passthrough()` |
| Append-only traffic log w/ batched flush | `bolog()` (separate `BLOCKOPS_LOG_FILE`) |
| **Default alias map** (`MODEL_ALIASES`, studio's ~8: `qwen35`, `coder`, `reason`, `r1`, …) | `_default_walter_aliases()` / `model_aliases:` config |
| **Default per-model backend map** (`MODEL_BACKENDS`) | `_default_walter_backends()` / `model_backends:` config |
| **`normalize_upstream_model()`** (alias OR unknown id -> canonical path; avoids cold model swap) | `normalize_upstream_model()` (verbatim); called in BOTH walter paths |
| **`CODER_BACKEND` (:8083) / `REASON_BACKEND` (:8084)** per-role routing | `CODER_BACKEND` / `REASON_BACKEND` consts -> `_backend_for_model()` |

## Parity GAPS / risks found

- **GAP-1 (RESOLVED) — walter mode is now a strict SUPERSET of BOTH blockops
  forks.** The studio's live `~alfredpennyworth/blockops-proxy.py` is a
  more-developed fork than the local/public `~/code/blockops-proxy/blockops-proxy.py`.
  Walter mode now folds in every studio-only feature while preserving laptop
  pass-through behaviour by default-on-empty-map. Proof matrix:

  | Feature | Public/laptop blockops | Studio blockops | gateway walter |
  |---|---|---|---|
  | 5-format tool-call parse + recovery | yes | yes | yes (`parse_tool_calls`) |
  | `<think>` / "Thinking Process:" strip | yes | yes | yes (`strip_thinking`) |
  | loop / cycle / session guards | yes | yes | yes (verbatim) |
  | context truncate + memory tiers | yes | yes | yes |
  | KV-cache purge + idle janitor | yes | yes | yes |
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
- **GAP-2 (MED) — studio chain currently has blockops *between* tracing and the
  model.** Live studio order is `alfred-llm-proxy:8091 -> metrics-proxy:8090 ->
  blockops:8080 -> mlx-vlm:8082`, and `qwen-397b`'s backend base_url is
  `:8090`. Running the gateway in `alfred` mode replaces 8091 **and** 8090
  (routing + tracing) but **NOT** the studio blockops at :8080. So the default
  `qwen-397b` base_url (`http://127.0.0.1:8090/v1`) must be **repointed to the
  studio blockops at `http://127.0.0.1:8080/v1`** at cutover, otherwise the
  gateway would call its own old tracing port. See cutover step 4. (Fully
  collapsing studio blockops into the gateway is out of scope here — see GAP-1.)
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

## POST-EVAL cutover checklist

Do all of this only after the eval finishes and you intend to bring the unified
gateway live. Build artifacts already staged at `~/gordon-gateway/` on the studio.

### A. Studio (alfred mode) — replaces llm-proxy (:8091) + metrics-proxy (:8090)

1. **Deps**: confirm the pyenv has everything (already verified):
   `~/.pyenv/versions/3.12.13/bin/python3 -c "import fastapi,uvicorn,httpx,opentelemetry,psutil"`.
   (Optional: `pip install pyyaml` if you want to use `config.yaml`; otherwise
   use env vars in the plist.)
2. **Config**: either copy `config.example.yaml` -> `~/gordon-gateway/config.yaml`
   with `mode: alfred`, or rely on env in the plist (set `GATEWAY_MODE=alfred`).
3. **Bring up the otel-collector first.** It's currently disabled
   (`ai.alfred.otel-collector.plist.disabled.20260601`). Re-enable it (and its
   downstream Tempo/Loki/Alloy on k3s per the LGTM migration) so spans have a
   sink. Verify `nc -z 127.0.0.1 4317`.
4. **Repoint the qwen-397b backend (GAP-2).** In `config.yaml` `backends:` (or
   leave default and instead point the gateway's listen at 8091 while keeping
   studio blockops at 8080), set `qwen-397b.base_url` to the **studio blockops**
   `http://127.0.0.1:8080/v1` — NOT `:8090`, which was the old tracing port that
   this gateway now *is*. Keep studio blockops (`ai.alfred.mlx-vlm-proxy.plist`,
   runs `~/blockops-proxy.py` on :8080) running unchanged.
5. **Swap the launchd plists** (do NOT delete the originals — rename to
   `.bak.PRE-GATEWAY`):
   - `ai.alfred.llm-proxy.plist`: change `ProgramArguments` to
     `~/.pyenv/versions/3.12.13/bin/python3  ~/gordon-gateway/gateway.py`;
     keep `EnvironmentVariables`: `MASTER_KEY=sk-local-macstudio`,
     `OPENROUTER_API_KEY=...`, `PORT=8091`, add `GATEWAY_MODE=alfred`,
     `OTEL_ENDPOINT=http://127.0.0.1:4317`. `WorkingDirectory ~/gordon-gateway`.
   - `ai.alfred.metrics-proxy.plist.disabled.20260531`: **leave disabled** — the
     gateway absorbs it. (Its old `metrics_state.json` lives in `~/alfred-otel/`;
     optionally copy it to `~/gordon-gateway/metrics_state.json` to carry the
     lifetime counters forward before first boot.)
   - `launchctl bootout gui/$(id -u)/ai.alfred.metrics-proxy` if still loaded;
     `launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/ai.alfred.llm-proxy.plist`
     to reload the rewritten plist (or `kickstart -k`).
6. **Client base_urls / ports — NO CHANGE NEEDED** for downstream clients:
   the gateway listens on the same `:8091` Hermes/claude-local already use, and
   serves the same `/v1/chat/completions`, `/v1/messages`,
   `/v1/messages/count_tokens`, `/v1/models`. The ONLY repoint is the internal
   one in step 4 (gateway's own qwen-397b backend 8090 -> 8080).
7. **Verify**:
   - `curl -s localhost:8091/health` -> `{"status":"ok","mode":"alfred","tracing":true}`
   - `curl -s localhost:8091/health/deep` -> per-backend ok/degraded
   - auth'd `POST /v1/chat/completions` (qwen-397b) and `POST /v1/messages`
     (stream + non-stream) return well-formed output.
   - **traces end-to-end**: hit the gateway, then check the otel-collector
     received `chat` spans (collector debug exporter log, or Tempo/Grafana
     "chat" service `llm-gateway`); confirm span attrs
     `gen_ai.usage.input_tokens/output_tokens`, `gen_ai.response.tool_calls`,
     `hermes.user_class`, and that `llm.requests` / `llm.cost.sonnet_equivalent`
     metrics increment in Prometheus.
   - `tail -f ~/.hermes/logs/gateway.log` shows `metrics: ...` lines.

### B. Laptop (walter mode) — replaces local blockops-proxy

1. Stage `gateway.py` + `config.example.yaml` to the MacBook (e.g.
   `~/code/gordon-gateway/`). Ensure `pip install fastapi uvicorn httpx
   opentelemetry-sdk opentelemetry-exporter-otlp psutil` (psutil + aiohttp were
   the only blockops deps; we drop aiohttp, add fastapi/uvicorn/httpx/otel).
2. Config: `mode: walter`, `walter_backend: http://127.0.0.1:8082` (or wherever
   Walter's mlx server listens), `model_name:` for the banner. Set
   `TRACING_ENABLED=false` unless a collector is reachable from the laptop
   (GAP-4).
3. If/when there's a launchd job for the laptop proxy, point it at
   `gateway.py` with `GATEWAY_MODE=walter`, `PORT=8080`. (No existing laptop
   plist was found; blockops there is run manually / by the Walter harness — just
   start `python3 gateway.py` in its place.)
4. Verify: `curl -s localhost:8080/health` -> `mode:walter`; a tool-calling
   request returns OpenAI `tool_calls`; `<think>` stripped; loop guard fires on a
   repeated tool call; `~/.blockops/proxy.log` shows the familiar
   `INBOUND/TOOL_CALLS/INFLIGHT_*` lines.

### Rollback
Re-enable the three original plists (rename `.bak.PRE-GATEWAY` back, re-enable
`metrics-proxy` and `otel-collector`), `bootout` the gateway, `bootstrap` the
originals. The original three source files were never modified.

"""
gordon-gateway — Unified LLM gateway proxy.

Single config-driven app that replaces three previously-separate proxies with
ONE codebase running a modular middleware chain:

    route (model selection)
      -> translate-request   (blockops tool-prompt / Anthropic->OpenAI)
        -> trace-start       (OTLP "chat" span)
          -> forward         (httpx to backend, with fallback chain)
        -> translate-response (blockops tool-call parsing / OpenAI->Anthropic)
      -> trace-end           (span end: token + latency + tool-call attrs)

Two operating MODES selected by env GATEWAY_MODE (or config `mode:`):

  * "alfred"  — Multi-backend routing + model selection + Anthropic Messages
                translation + OTLP tracing. NO blockops tool-call translation
                or loop guards. Listen :8091 by default.

  * "walter"  — Single/role-routed local backend with prompt-injected blockops
                tool-call translation + loop guards + context truncation +
                memory warnings + idle KV-cache janitor + OTLP tracing.
                Listen :8080 by default.

Framework: FastAPI + uvicorn + httpx (with optional PyYAML, psutil, and
OpenTelemetry). See pyproject.toml / requirements.txt for tested ranges.

CONFIG: see config.example.yaml. If PyYAML is unavailable or no config file is
present, every setting falls back to an environment variable (documented in the
README), so the gateway runs with zero config files in either mode.
"""

from __future__ import annotations

import asyncio
import datetime
import hashlib
import hmac
import ipaddress
import json
import logging
import logging.config
import os
import re
import time
import urllib.parse
import uuid
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, AsyncIterator

import httpx
import uvicorn
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

# psutil is only needed for walter-mode memory warnings; import lazily-safe.
try:
    import psutil
except Exception:  # pragma: no cover
    psutil = None

# OpenTelemetry is optional: if tracing is disabled or libs missing, the trace
# middleware degrades to no-ops so the gateway still serves traffic.
_OTEL_OK = True
try:
    from opentelemetry import metrics as otel_metrics
    from opentelemetry import trace as otel_trace
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
except Exception:  # pragma: no cover
    _OTEL_OK = False


# ===========================================================================
# CONFIG
# ===========================================================================
_SECRET_ENV_NAMES = frozenset({"MASTER_KEY", "OPENROUTER_API_KEY"})


def _load_secret_env() -> None:
    """Load only gateway secrets from a deployment-local, owner-only file.

    launchd property lists are ordinary files and must not contain credentials.
    Explicit process environment variables still win, which keeps tests and
    one-off deployments straightforward.
    """
    path = Path(os.environ.get(
        "GATEWAY_SECRET_ENV", os.path.expanduser("~/.config/gordon-gateway/secrets.env")
    ))
    if not path.exists():
        return
    stat_result = path.stat()
    if stat_result.st_uid != os.getuid() or stat_result.st_mode & 0o077:
        raise SystemExit(f"gateway secret file must be owner-only: {path}")
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key not in _SECRET_ENV_NAMES or key in os.environ:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if value:
            os.environ[key] = value


_load_secret_env()


def _load_config() -> dict:
    """Layered config: YAML file (if present + PyYAML available) merged under
    environment-variable overrides. Returns a flat dict of settings."""
    cfg: dict[str, Any] = {}
    cfg_path = os.environ.get("GATEWAY_CONFIG", os.path.expanduser("~/gordon-gateway/config.yaml"))
    if os.path.exists(cfg_path):
        try:
            import yaml  # type: ignore
            with open(cfg_path) as fh:
                loaded = yaml.safe_load(fh) or {}
            if isinstance(loaded, dict):
                cfg.update(loaded)
        except Exception as e:  # pragma: no cover
            print(f"[gateway] WARN failed to load config {cfg_path}: {e}")
    return cfg


_CFG = _load_config()


def _get(key: str, env: str, default: Any) -> Any:
    """Resolve a setting: env var wins, then YAML, then default."""
    if env in os.environ:
        return os.environ[env]
    if key in _CFG:
        return _CFG[key]
    return default


def _get_bool(key: str, env: str, default: bool) -> bool:
    v = _get(key, env, default)
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def _parse_strict_bool(value: Any, *, field_name: str) -> bool:
    """Require an actual Python bool; reject string booleans like 'false' and ints."""
    if isinstance(value, bool):
        return value
    raise ValueError(
        f"{field_name} must be a boolean (true/false), got {type(value).__name__}: {value!r}"
    )


def _get_strict_bool(key: str, env: str, default: bool) -> bool:
    """Resolve a security-sensitive boolean setting without truthy-string coercion.

    Environment variables accept only explicit boolean literals; YAML config
    values must be native booleans (quoted 'false'/'true' strings are rejected).
    """
    if env in os.environ:
        raw = os.environ[env].strip().lower()
        if raw in ("1", "true", "yes", "on"):
            return True
        if raw in ("0", "false", "no", "off", ""):
            return False
        raise ValueError(f"{env} must be a boolean value, got {os.environ[env]!r}")
    if key in _CFG:
        return _parse_strict_bool(_CFG[key], field_name=key)
    return default


def _get_int(key: str, env: str, default: int) -> int:
    try:
        return int(_get(key, env, default))
    except (TypeError, ValueError):
        return default


def _get_float(key: str, env: str, default: float) -> float:
    try:
        return float(_get(key, env, default))
    except (TypeError, ValueError):
        return default


MODE = str(_get("mode", "GATEWAY_MODE", "alfred")).strip().lower()
if MODE not in ("alfred", "walter"):
    raise SystemExit(f"GATEWAY_MODE must be 'alfred' or 'walter', got {MODE!r}")

# Default listen port differs per mode (alfred=8091, walter=8080) so dropping
# the gateway in as a direct replacement needs no port change.
_DEFAULT_PORT = 8091 if MODE == "alfred" else 8080
PORT = _get_int("port", "PORT", _DEFAULT_PORT)
HOST = str(_get("host", "HOST", "127.0.0.1"))

# Tracing applies in BOTH modes. Toggle with TRACING_ENABLED.
# Default is metadata-only (TRACE_CONTENT=false): prompt/completion text is NOT
# exported to OTLP unless explicitly opted in.
TRACING_ENABLED = _get_bool("tracing_enabled", "TRACING_ENABLED", True) and _OTEL_OK
OTEL_ENDPOINT = str(_get("otel_endpoint", "OTEL_ENDPOINT", "http://127.0.0.1:4317"))
OTEL_HEADERS_RAW = str(_get("otel_headers", "OTEL_EXPORTER_OTLP_HEADERS", ""))
TRACE_CONTENT = _get_bool("trace_content", "TRACE_CONTENT", False)
SERVICE_NAME = str(_get("service_name", "OTEL_SERVICE_NAME", "llm-gateway"))

# Cost + attribution config (used by trace + metrics layer)
SONNET_INPUT_PRICE = _get_float("sonnet_input_price", "SONNET_INPUT_PRICE", 15.0) / 1_000_000
SONNET_OUTPUT_PRICE = _get_float("sonnet_output_price", "SONNET_OUTPUT_PRICE", 75.0) / 1_000_000
PRIMARY_USER_ID = str(_get("primary_user_id", "PRIMARY_USER_ID", "primary-user"))
STATE_FILE = Path(str(_get(
    "metrics_state_file", "METRICS_STATE_FILE",
    os.path.expanduser("~/gordon-gateway/metrics_state.json"),
)))
STATE_SAVE_INTERVAL = _get_int("metrics_state_save_interval", "METRICS_STATE_SAVE_INTERVAL", 30)

# ---- alfred-mode: auth + routing ----
MASTER_KEY = _get("master_key", "MASTER_KEY", None)
if MODE == "alfred" and not MASTER_KEY:
    raise SystemExit("MASTER_KEY required in alfred mode")

MAX_BODY_BYTES = _get_int("max_body_bytes", "MAX_BODY_BYTES", 50 * 1024 * 1024)
DEFAULT_MAX_TOKENS = _get_int("default_max_tokens", "DEFAULT_MAX_TOKENS", 16384)


def _resolve_max_tokens(
    raw_value: Any,
    default: int | None = None,
    ceiling: int | None = None,
) -> int:
    """Resolve max_tokens without inflating explicit caller budgets.

    Applies the default only when max_tokens is omitted, None, or non-positive.
    Any explicit positive budget (including small values like 128 or 1024) is
    preserved verbatim. If a positive ceiling is configured, only values above
    the ceiling are clamped down.
    """
    fallback = DEFAULT_MAX_TOKENS if default is None else default
    resolved: int | None = None
    if raw_value is not None and not isinstance(raw_value, bool):
        try:
            parsed = int(raw_value)
        except (TypeError, ValueError):
            parsed = 0
        if parsed > 0:
            resolved = parsed
    if resolved is None:
        resolved = fallback
    if ceiling is not None and ceiling > 0 and resolved > ceiling:
        return ceiling
    return resolved

HEADER_BLOCKLIST = {
    "authorization", "x-api-key",
    "host", "content-length", "connection", "transfer-encoding",
    "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailer", "upgrade",
    "accept-encoding",
    "cookie", "set-cookie", "referer", "user-agent",
    "forwarded", "x-forwarded-for", "x-forwarded-host", "x-forwarded-proto",
    "x-real-ip", "via",
}

DEFAULT_TIMEOUT = httpx.Timeout(connect=10.0, read=600.0, write=60.0, pool=10.0)
HEALTH_TIMEOUT = httpx.Timeout(5.0)


def _is_loopback_host(host: str | None) -> bool:
    if not host:
        return False
    cleaned = host.strip().lower().strip("[]")
    if cleaned in ("localhost", "127.0.0.1", "::1"):
        return True
    try:
        return ipaddress.ip_address(cleaned).is_loopback
    except ValueError:
        return False


VALID_TRUST_BOUNDARIES: frozenset[str] = frozenset({"local", "external"})


def _infer_trust_boundary(base_url: str) -> str:
    parsed = urllib.parse.urlparse(base_url if "://" in base_url else f"http://{base_url}")
    return "local" if _is_loopback_host(parsed.hostname) else "external"


def _validate_trust_boundary(raw: Any, base_url: str) -> str:
    """Validate and canonicalize a backend trust_boundary value.

    Empty/omitted values are inferred from base_url ('local' for loopback,
    'external' otherwise). Explicit values must be strings matching the
    canonical enum ('local' or 'external', case-insensitive); any other type
    or unknown label fails closed with ValueError.
    """
    if raw is None or raw == "":
        return _infer_trust_boundary(base_url)
    if not isinstance(raw, str):
        raise ValueError(
            f"trust_boundary must be a string ('local' or 'external'), got {type(raw).__name__}"
        )
    normalized = raw.strip().lower()
    if normalized not in VALID_TRUST_BOUNDARIES:
        raise ValueError(
            f"invalid trust_boundary {raw!r}: must be 'local' or 'external'"
        )
    return normalized


def _parse_status_code_set(raw: Any) -> frozenset[int]:
    if raw is None or raw == "":
        return frozenset()
    items = raw.split(",") if isinstance(raw, str) else list(raw) if isinstance(raw, (list, tuple, set, frozenset)) else [raw]
    codes: set[int] = set()
    for item in items:
        s = str(item).strip()
        if not s:
            continue
        try:
            code = int(s)
        except ValueError:
            continue
        if 400 <= code < 500:
            codes.add(code)
    return frozenset(codes)


def _parse_string_set(raw: Any) -> frozenset[str]:
    if raw is None or raw == "":
        return frozenset()
    items = raw.split(",") if isinstance(raw, str) else list(raw) if isinstance(raw, (list, tuple, set, frozenset)) else [raw]
    return frozenset(str(x).strip() for x in items if str(x).strip())


# Retry & cross-boundary fallback policy:
# - Transport/timeout/protocol failures and HTTP 5xx are retryable across the chain.
# - HTTP 4xx statuses are TERMINAL by default unless explicitly listed in
#   RETRYABLE_4XX_STATUSES / retryable_4xx_statuses.
# - Crossing from a local backend to an external/cloud fallback backend requires
#   explicit opt-in via ALLOW_EXTERNAL_FALLBACK=true, EXTERNAL_FALLBACK_ROUTES,
#   or backend.allow_external_fallback=True.
RETRYABLE_4XX_STATUSES: frozenset[int] = _parse_status_code_set(
    _get("retryable_4xx_statuses", "RETRYABLE_4XX_STATUSES", "")
)
ALLOW_EXTERNAL_FALLBACK: bool = _get_strict_bool(
    "allow_external_fallback", "ALLOW_EXTERNAL_FALLBACK", False
)
EXTERNAL_FALLBACK_ROUTES: frozenset[str] = _parse_string_set(
    _get("external_fallback_routes", "EXTERNAL_FALLBACK_ROUTES", "")
)


def is_retryable_status(
    status_code: int,
    retryable_4xx: frozenset[int] | set[int] | None = None,
) -> bool:
    """Return True if an upstream HTTP status should advance the fallback chain.

    HTTP 5xx statuses are retryable by default. HTTP 4xx statuses are terminal
    by default unless explicitly opted in via RETRYABLE_4XX_STATUSES.
    """
    if status_code >= 500:
        return True
    allowed = RETRYABLE_4XX_STATUSES if retryable_4xx is None else retryable_4xx
    return 400 <= status_code < 500 and status_code in allowed


@dataclass
class Backend:
    name: str
    base_url: str
    upstream_model: str
    api_key: str
    extra_headers: dict[str, str] = field(default_factory=dict)
    trust_boundary: str = ""
    allow_external_fallback: bool = False

    def __post_init__(self) -> None:
        self.trust_boundary = _validate_trust_boundary(self.trust_boundary, self.base_url)
        self.allow_external_fallback = _parse_strict_bool(
            self.allow_external_fallback,
            field_name=f"backend {self.name!r} allow_external_fallback",
        )


QWEN397B_MODEL_PATH = str(_get("qwen397b_model_path", "QWEN397B_MODEL_PATH", "qwen3.5-397b-6bit"))
ROUTER_BASE_URL = str(_get("router_base_url", "ROUTER_BASE_URL", "http://127.0.0.1:9003/v1"))
OPENROUTER_REFERER = str(_get("openrouter_referer", "OPENROUTER_REFERER", "http://localhost"))


def _default_alfred_backends() -> dict[str, Backend]:
    """Generic default backend table for alfred mode. Overridable via config
    `backends:` (list of dicts) or deployment-local environment variables."""
    return {
        "qwen-397b": Backend(
            name="qwen-397b",
            base_url="http://127.0.0.1:8090/v1",
            upstream_model=QWEN397B_MODEL_PATH,
            api_key="not-needed",
            trust_boundary="local",
        ),
        "kimi-k2.6": Backend(
            name="kimi-k2.6",
            base_url="http://127.0.0.1:8082/v1",
            upstream_model="kimi-k2.6",
            api_key="not-needed",
            trust_boundary="local",
        ),
        "glm-5.1": Backend(
            name="glm-5.1",
            base_url="http://127.0.0.1:8082/v1",
            upstream_model="glm-5.1",
            api_key="not-needed",
            trust_boundary="local",
        ),
        "glm-5.1-test": Backend(
            name="glm-5.1-test",
            base_url="http://127.0.0.1:8083/v1",
            upstream_model="glm-5.1",
            api_key="not-needed",
            trust_boundary="local",
        ),
        "qwen-alfred-router": Backend(
            name="qwen-alfred-router",
            base_url=ROUTER_BASE_URL,
            upstream_model="qwen-alfred-router",
            api_key="not-needed",
        ),
        "openrouter-free": Backend(
            name="openrouter-free",
            base_url="https://openrouter.ai/api/v1",
            upstream_model="nvidia/nemotron-3-super-120b-a12b:free",
            api_key=os.environ.get("OPENROUTER_API_KEY", ""),
            extra_headers={
                "HTTP-Referer": OPENROUTER_REFERER,
                "X-Title": "gordon-gateway",
            },
            trust_boundary="external",
        ),
    }


def _backends_from_config() -> dict[str, Backend]:
    raw = _CFG.get("backends")
    if raw is None:
        return _default_alfred_backends()
    if not isinstance(raw, list):
        raise ValueError(f"backends config must be a list, got {type(raw).__name__}")
    out: dict[str, Backend] = {}
    for idx, b in enumerate(raw):
        if not isinstance(b, dict):
            raise ValueError(f"backends[{idx}] must be a mapping, got {type(b).__name__}")
        name = b.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"backends[{idx}] requires a non-empty string 'name'")
        base_url = b.get("base_url")
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError(f"backend {name!r} requires a non-empty string 'base_url'")
        api_key = b.get("api_key", "not-needed")
        if isinstance(api_key, str) and api_key.startswith("env:"):
            api_key = os.environ.get(api_key[4:], "")
        trust_boundary = _validate_trust_boundary(b.get("trust_boundary"), base_url)
        if "allow_external_fallback" in b:
            allow_external = _parse_strict_bool(
                b["allow_external_fallback"],
                field_name=f"backend {name!r} allow_external_fallback",
            )
        else:
            allow_external = False
        out[name] = Backend(
            name=name,
            base_url=base_url,
            upstream_model=b.get("upstream_model", name),
            api_key=api_key,
            extra_headers=b.get("extra_headers", {}) or {},
            trust_boundary=trust_boundary,
            allow_external_fallback=allow_external,
        )
    return out or _default_alfred_backends()


BACKENDS: dict[str, Backend] = _backends_from_config() if MODE == "alfred" else {}

# Default fallback map is empty so local traffic never silently crosses a cloud
# boundary. Configure `fallbacks:` in config.yaml and authorize external hops
# explicitly via `allow_external_fallback` / `external_fallback_routes`.
FALLBACKS: dict[str, list[str]] = _CFG.get("fallbacks") if isinstance(_CFG.get("fallbacks"), dict) else {}

SWAP_IN_IDENTITY: dict[str, tuple[str, ...]] = {
    "kimi-k2.6": ("Kimi", "kimi"),
    "glm-5.1": ("GLM", "glm"),
}
_IDENTITY_CACHE: dict[str, tuple[float, bool]] = {}
_IDENTITY_CACHE_TTL = 5.0

# ---- walter-mode: blockops backend + tuning ----
WALTER_BACKEND = str(_get("walter_backend", "BACKEND_URL", "http://127.0.0.1:8082"))
BACKEND_STRIP_PATH_PREFIX = str(_get("backend_strip_path_prefix", "BACKEND_STRIP_PATH_PREFIX", ""))
_MODEL_NAME = str(_get("model_name", "MODEL_NAME", ""))
_MODEL_DISPLAY = _MODEL_NAME or "local model"
# walter canonical model id: an unknown/missing model id is coerced to this so
# an OpenAI-compatible backend does not cache-clear / cold-reload on an alias
# mismatch. Env/YAML overridable.
WALTER_CANONICAL_MODEL = str(_get(
    "walter_canonical_model", "WALTER_CANONICAL_MODEL",
    "qwen3.5-122b-a10b-8bit",
))

# Per-role backend overrides. Default to WALTER_BACKEND so a single-server host
# keeps single-backend behaviour; multi-server hosts can point these elsewhere.
CODER_BACKEND = str(_get("coder_backend", "CODER_BACKEND", WALTER_BACKEND))
REASON_BACKEND = str(_get("reason_backend", "REASON_BACKEND", WALTER_BACKEND))


def _default_walter_aliases() -> dict[str, str]:
    """Default walter MODEL_ALIASES (~8 aliases). Each maps a friendly alias to
    the canonical backend model id so the served-model-name matches what the
    backend expects."""
    canon = WALTER_CANONICAL_MODEL
    return {
        "qwen35": canon,
        "qwen35-base": canon,
        "qwen-coder": canon,
        "coder": canon,
        "qwen-reason": canon,
        "reason": canon,
        "r1": canon,
        "mistral": canon,
    }


def _default_walter_backends() -> dict[str, str]:
    """Per-model backend map (studio parity). coder -> CODER_BACKEND,
    reason/r1 -> REASON_BACKEND, everything else -> WALTER_BACKEND. Keyed by both
    the alias and the canonical path so a request that already carries the path
    still routes. On the laptop CODER/REASON default to WALTER_BACKEND, so this
    collapses to single-backend behaviour (laptop parity)."""
    return {
        WALTER_CANONICAL_MODEL: WALTER_BACKEND,
        "qwen35": WALTER_BACKEND,
        "qwen35-base": WALTER_BACKEND,
        "qwen-coder": CODER_BACKEND,
        "coder": CODER_BACKEND,
        "qwen-reason": REASON_BACKEND,
        "reason": REASON_BACKEND,
        "r1": REASON_BACKEND,
        "mistral": WALTER_BACKEND,
    }


# YAML/env (`model_aliases:` / `model_backends:`) WIN; otherwise the studio
# defaults apply. An explicitly-provided empty map ({}) is honoured as a way to
# force pure public/laptop pass-through (no aliasing) if ever desired.
MODEL_ALIASES: dict = _CFG.get("model_aliases") if isinstance(_CFG.get("model_aliases"), dict) else _default_walter_aliases()
MODEL_BACKENDS: dict = _CFG.get("model_backends") if isinstance(_CFG.get("model_backends"), dict) else _default_walter_backends()
ALIAS_REVERSE = {v: k for k, v in MODEL_ALIASES.items()}


def normalize_upstream_model(data: dict, log_fn=None) -> None:
    """Coerce data['model'] to the configured backend's canonical path so unknown
    names don't trigger a cache-clear/cold-reload on the server. Ported from the
    studio's blockops-proxy.py.

      - model is an alias key (e.g. "qwen35")  -> rewrite to canonical path
      - model is already a canonical path      -> leave as-is
      - model missing or unrecognized          -> coerce to qwen35's path

    No-op when MODEL_ALIASES is empty (preserves public/laptop pass-through
    parity: with no alias map there is nothing to coerce to)."""
    if not MODEL_ALIASES:
        return
    original = data.get("model", "<none>")
    if original in MODEL_ALIASES:
        data["model"] = MODEL_ALIASES[original]
        if log_fn:
            log_fn(f"MODEL_ALIAS {original} → {data['model']}")
    elif original in MODEL_ALIASES.values():
        return
    else:
        default_path = MODEL_ALIASES.get("qwen35") or WALTER_CANONICAL_MODEL
        if default_path:
            data["model"] = default_path
            if log_fn:
                log_fn(f"MODEL_COERCED unrecognized={original} → {default_path}")

HEARTBEAT_INTERVAL = _get_int("heartbeat_interval", "HEARTBEAT_INTERVAL", 5)
MAX_MESSAGES = _get_int("max_messages", "MAX_MESSAGES", 300)
LOOP_DETECT_NUDGE_THRESHOLD = _get_int("loop_nudge_threshold", "LOOP_NUDGE_THRESHOLD", 4)
LOOP_DETECT_STOP_THRESHOLD = _get_int("loop_stop_threshold", "LOOP_STOP_THRESHOLD", 8)
LOOP_DETECT_CYCLE_NUDGE_REPS = _get_int("loop_cycle_nudge_reps", "LOOP_CYCLE_NUDGE_REPS", 3)
LOOP_DETECT_CYCLE_STOP_REPS = _get_int("loop_cycle_stop_reps", "LOOP_CYCLE_STOP_REPS", 4)
LOOP_DETECT_CYCLE_MAX_LEN = _get_int("loop_cycle_max_len", "LOOP_CYCLE_MAX_LEN", 4)
LOOP_DETECT_SESSION_NUDGE_LIMIT = _get_int("loop_session_nudge_limit", "LOOP_SESSION_NUDGE_LIMIT", 8)
TOOL_RESULT_COMPRESSION_THRESHOLD = _get_int("tool_result_compression_threshold", "TOOL_RESULT_COMPRESSION_THRESHOLD", 15_000)
TOKEN_WARN_THRESHOLD = _get_int("token_warn_threshold", "TOKEN_WARN_THRESHOLD", 80_000)
TOKEN_HARD_THRESHOLD = _get_int("token_hard_threshold", "TOKEN_HARD_THRESHOLD", 100_000)
REPEAT_PARSE_ERROR_STOP = _get_int("repeat_parse_error_stop", "REPEAT_PARSE_ERROR_STOP", 2)
MAX_CONCURRENT = _get_int("max_concurrent", "MAX_CONCURRENT", 3)
COLD_START_WARN_S = _get_int("cold_start_warn_s", "COLD_START_WARN_S", 300)
CACHE_PURGE_MB_THRESHOLD = _get_int("cache_purge_mb_threshold", "CACHE_PURGE_MB_THRESHOLD", 12_000)
CACHE_IDLE_SECONDS = _get_int("cache_idle_seconds", "CACHE_IDLE_SECONDS", 60)
SESSION_STATE_MAX = max(1, _get_int("session_state_max", "SESSION_STATE_MAX", 512))
SESSION_STATE_TTL_SECONDS = max(60, _get_int(
    "session_state_ttl_seconds", "SESSION_STATE_TTL_SECONDS", 12 * 60 * 60,
))
SESSION_PARSE_ERROR_KEYS_MAX = max(1, _get_int(
    "session_parse_error_keys_max", "SESSION_PARSE_ERROR_KEYS_MAX", 32,
))

BLOCKOPS_LOG_FILE = str(_get("blockops_log_file", "BLOCKOPS_LOG_FILE", os.path.expanduser("~/.gordon-gateway/blockops.log")))


# ===========================================================================
# LOGGING
# ===========================================================================
LOG_FILE = str(_get("log_file", "LOG_FILE", os.path.expanduser("~/.gordon-gateway/logs/gateway.log")))
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
LOG_CONFIG: dict[str, Any] = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {"default": {"format": "%(asctime)s %(levelname)s %(name)s %(message)s"}},
    "handlers": {
        "rotating": {
            "()": RotatingFileHandler,
            "filename": LOG_FILE,
            "maxBytes": 10 * 1024 * 1024,
            "backupCount": 5,
            "formatter": "default",
        },
    },
    "loggers": {
        "gordon-gateway": {"handlers": ["rotating"], "level": "INFO", "propagate": False},
        "uvicorn": {"handlers": ["rotating"], "level": "INFO", "propagate": False},
        "uvicorn.error": {"handlers": ["rotating"], "level": "INFO", "propagate": False},
        "uvicorn.access": {"handlers": ["rotating"], "level": "INFO", "propagate": False},
    },
    "root": {"handlers": ["rotating"], "level": "INFO"},
}
logging.config.dictConfig(LOG_CONFIG)
log = logging.getLogger("gordon-gateway")

# blockops keeps its own append-only traffic log (separate from the rotating
# app log) for parity with the original BLOCKOPS_LOG_FILE behaviour.
_bo_fh = None
_bo_pending = 0
_BO_FLUSH_EVERY = 20


def bolog(msg: str) -> None:
    """blockops traffic logger (walter mode). Mirrors blockops-proxy.log()."""
    global _bo_fh, _bo_pending
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        if _bo_fh is None or _bo_fh.closed:
            os.makedirs(os.path.dirname(BLOCKOPS_LOG_FILE), exist_ok=True)
            _bo_fh = open(BLOCKOPS_LOG_FILE, "a")
        _bo_fh.write(f"[{ts}] {msg}\n")
        _bo_pending += 1
        if (_bo_pending >= _BO_FLUSH_EVERY or any(
            tag in msg for tag in (
                "ERROR", "CRITICAL", "BACKEND_ERROR", "LOOP_DETECTED", "LOOP_NUDGE",
                "LOOP_CYCLE_DETECTED", "LOOP_CYCLE_NUDGE", "LOOP_SESSION_STOP",
                "SESSION_NUDGE_COUNT",
            )
        )):
            _bo_fh.flush()
            _bo_pending = 0
    except Exception:
        _bo_fh = None
        _bo_pending = 0


# ===========================================================================
# OTLP TRACING + METRICS LAYER  (from metrics-proxy)
# ===========================================================================
_tracer = None
_meter = None
_metric_objs: dict[str, Any] = {}
_trace_provider = None
_metric_provider = None
_accum: dict[str, float] = {}


def _parse_otel_headers(raw: str) -> dict[str, str]:
    headers: dict[str, str] = {}
    for part in (raw or "").split(","):
        item = part.strip()
        if not item or "=" not in item:
            continue
        k, v = item.split("=", 1)
        k = k.strip()
        v = v.strip()
        if k and v:
            headers[k] = v
    return headers


def _validate_otel_transport(endpoint: str, headers_raw: str = "") -> tuple[bool, dict[str, str]]:
    """Validate OTLP collector transport security.

    Loopback collectors (127.0.0.1 / localhost / ::1) may use plaintext HTTP or
    HTTPS. Any non-loopback collector MUST use https:// TLS (insecure=False) and
    carry explicit authentication headers via OTEL_EXPORTER_OTLP_HEADERS.
    """
    ep = (endpoint or "").strip()
    if not ep:
        raise ValueError("OTEL_ENDPOINT must not be empty when tracing is enabled")
    has_scheme = "://" in ep
    parsed = urllib.parse.urlparse(ep if has_scheme else f"http://{ep}")
    scheme = parsed.scheme.lower() if has_scheme else "http"
    headers = _parse_otel_headers(headers_raw)
    if _is_loopback_host(parsed.hostname):
        insecure = scheme != "https"
        return insecure, headers
    if scheme != "https":
        raise ValueError(
            f"Non-loopback OTLP endpoint requires https:// TLS: {ep!r}"
        )
    auth_keys = {k.lower() for k in headers}
    if not (auth_keys & {"authorization", "x-api-key", "api-key", "x-otlp-token"}):
        raise ValueError(
            f"Non-loopback OTLP endpoint requires authentication headers via "
            f"OTEL_EXPORTER_OTLP_HEADERS (e.g. authorization=Bearer ...): {ep!r}"
        )
    return False, headers


def _init_otel() -> None:
    global _tracer, _meter, _trace_provider, _metric_provider
    if not TRACING_ENABLED:
        return
    insecure, otel_headers = _validate_otel_transport(OTEL_ENDPOINT, OTEL_HEADERS_RAW)
    exporter_kwargs: dict[str, Any] = {"endpoint": OTEL_ENDPOINT, "insecure": insecure}
    if otel_headers:
        exporter_kwargs["headers"] = tuple(otel_headers.items())
    resource = Resource.create({"service.name": SERVICE_NAME})
    metric_exporter = OTLPMetricExporter(**exporter_kwargs)
    reader = PeriodicExportingMetricReader(metric_exporter, export_interval_millis=10_000)
    _metric_provider = MeterProvider(resource=resource, metric_readers=[reader])
    otel_metrics.set_meter_provider(_metric_provider)

    span_exporter = OTLPSpanExporter(**exporter_kwargs)
    _trace_provider = TracerProvider(resource=resource)
    _trace_provider.add_span_processor(BatchSpanProcessor(span_exporter))
    otel_trace.set_tracer_provider(_trace_provider)

    _tracer = otel_trace.get_tracer("llm.gateway", "1.0.0")
    _meter = otel_metrics.get_meter("llm.gateway", "1.0.0")
    _metric_objs["tokens_input"] = _meter.create_counter("llm.tokens.input", description="Total input tokens")
    _metric_objs["tokens_output"] = _meter.create_counter("llm.tokens.output", description="Total output tokens")
    _metric_objs["tokens_total"] = _meter.create_counter("llm.tokens.total", description="Total tokens (input + output)")
    _metric_objs["tool_calls"] = _meter.create_counter("llm.tool_calls", description="Number of tool calls")
    _metric_objs["requests"] = _meter.create_counter("llm.requests", description="Total requests")
    _metric_objs["duration"] = _meter.create_histogram("llm.request.duration", description="Request latency in seconds", unit="s")
    _metric_objs["cost"] = _meter.create_counter("llm.cost.sonnet_equivalent", description="Opus equivalent cost in USD", unit="USD")


def _resolve_user(headers: dict) -> tuple[str, str]:
    user_id = headers.get("x-hermes-user") or headers.get("X-Hermes-User") or ""
    if not user_id:
        return PRIMARY_USER_ID, "primary"  # single-user default: header-less traffic maps to PRIMARY_USER_ID
    if user_id == PRIMARY_USER_ID:
        return user_id, "primary"
    return user_id, "other"


def _load_state() -> None:
    global _accum
    try:
        if STATE_FILE.exists():
            _accum = json.loads(STATE_FILE.read_text())
            log.info("metrics state loaded: %d keys", len(_accum))
    except Exception as e:
        log.warning("failed to load metrics state: %s", e)
        _accum = {}


def _save_state() -> None:
    try:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(json.dumps(_accum))
    except Exception as e:
        log.warning("failed to save metrics state: %s", e)


def _restore_counters() -> None:
    if not TRACING_ENABLED:
        return
    for key, value in _accum.items():
        parts = key.rsplit("|", 1)
        if len(parts) != 2:
            continue
        metric_name, attrs_json = parts
        try:
            attrs = json.loads(attrs_json)
        except Exception:
            continue
        if metric_name == "cost" and value > 0:
            _metric_objs["cost"].add(value, attrs)
        elif metric_name == "tokens_in" and value > 0:
            _metric_objs["tokens_input"].add(int(value), attrs)
        elif metric_name == "tokens_out" and value > 0:
            _metric_objs["tokens_output"].add(int(value), attrs)
        elif metric_name == "tokens_total" and value > 0:
            _metric_objs["tokens_total"].add(int(value), attrs)
        elif metric_name == "requests" and value > 0:
            _metric_objs["requests"].add(int(value), attrs)


def _record_metrics(input_tokens: int, output_tokens: int, tool_call_count: int,
                    duration: float, model: str, path: str,
                    user_id: str = "", user_class: str = "unknown") -> None:
    if not TRACING_ENABLED:
        return
    attrs = {"model": model, "endpoint": path, "user_class": user_class, "user_id": user_id}
    attrs_json = json.dumps(attrs, sort_keys=True)
    _metric_objs["tokens_input"].add(input_tokens, attrs)
    _metric_objs["tokens_output"].add(output_tokens, attrs)
    _metric_objs["tokens_total"].add(input_tokens + output_tokens, attrs)
    _metric_objs["requests"].add(1, attrs)
    _metric_objs["duration"].record(duration, attrs)
    if tool_call_count > 0:
        _metric_objs["tool_calls"].add(tool_call_count, attrs)
    sonnet_cost = (input_tokens * SONNET_INPUT_PRICE) + (output_tokens * SONNET_OUTPUT_PRICE)
    _metric_objs["cost"].add(sonnet_cost, attrs)
    _accum[f"cost|{attrs_json}"] = _accum.get(f"cost|{attrs_json}", 0) + sonnet_cost
    _accum[f"tokens_in|{attrs_json}"] = _accum.get(f"tokens_in|{attrs_json}", 0) + input_tokens
    _accum[f"tokens_out|{attrs_json}"] = _accum.get(f"tokens_out|{attrs_json}", 0) + output_tokens
    _accum[f"tokens_total|{attrs_json}"] = _accum.get(f"tokens_total|{attrs_json}", 0) + input_tokens + output_tokens
    _accum[f"requests|{attrs_json}"] = _accum.get(f"requests|{attrs_json}", 0) + 1
    log.info("metrics: model=%s user=%s class=%s in=%d out=%d tools=%d dur=%.2fs cost=$%.6f",
             model, user_id or "-", user_class, input_tokens, output_tokens, tool_call_count, duration, sonnet_cost)


class _NullSpan:
    """No-op span used when tracing is disabled so call sites stay uniform."""
    def set_attribute(self, *a, **k): pass
    def add_event(self, *a, **k): pass
    def end(self, *a, **k): pass
    def __enter__(self): return self
    def __exit__(self, *a): return False


def _start_chat_span(model: str, streaming: bool, body: bytes,
                     user_id: str, user_class: str, current: bool):
    """trace-start middleware. Returns a span (real or null). When current=True
    the span is entered as the current span (non-streaming path); otherwise the
    caller ends it manually (streaming path)."""
    if not TRACING_ENABLED:
        return _NullSpan()
    if current:
        span = _tracer.start_span("chat", kind=otel_trace.SpanKind.SERVER)
    else:
        span = _tracer.start_span("chat", kind=otel_trace.SpanKind.SERVER)
    span.set_attribute("gen_ai.system", "openai")
    span.set_attribute("gen_ai.request.model", model)
    span.set_attribute("gen_ai.request.streaming", streaming)
    span.set_attribute("hermes.user_id", user_id)
    span.set_attribute("hermes.user_class", user_class)
    if TRACE_CONTENT:
        try:
            parsed = json.loads(body)
            span.add_event("gen_ai.content.prompt", {"gen_ai.prompt": _summarize_messages(parsed.get("messages", []))})
        except Exception:
            pass
    return span


def _summarize_messages(messages: list) -> str:
    parts = []
    for msg in messages:
        role = msg.get("role", "?")
        content = msg.get("content", "")
        if isinstance(content, list):
            text_parts = [c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text"]
            content = " ".join(text_parts)
        if isinstance(content, str) and len(content) > 500:
            content = content[:500] + "..."
        parts.append(f"{role}: {content}")
    return "\n".join(parts)


def _count_tool_calls_in_chunk(data: dict) -> int:
    count = 0
    for choice in data.get("choices", []):
        msg = choice.get("message", {})
        if msg.get("tool_calls"):
            count += len(msg["tool_calls"])
        delta = choice.get("delta", {})
        if delta.get("tool_calls"):
            count += len(delta["tool_calls"])
    return count


def _extract_usage(data: dict) -> tuple[int, int]:
    usage = data.get("usage", {}) or {}
    inp = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
    out = usage.get("completion_tokens") or usage.get("output_tokens") or 0
    return inp, out


# ===========================================================================
# APP
# ===========================================================================
app = FastAPI(title=f"gordon-gateway[{MODE}]", version="1.0.0")


@app.get("/health")
@app.get("/health/liveliness")
@app.get("/health/readiness")
async def health():
    return {"status": "ok", "mode": MODE, "tracing": TRACING_ENABLED}


@app.get("/health/deep")
async def health_deep():
    """Active backend health probe. alfred: pings each backend /v1/models.
    walter: pings the single backend root."""
    if MODE == "alfred":
        statuses: dict[str, str] = {}
        async with httpx.AsyncClient(timeout=HEALTH_TIMEOUT) as client:
            for name, backend in BACKENDS.items():
                url = f"{backend.base_url.rstrip('/')}/models"
                try:
                    r = await client.get(url, headers={"Authorization": f"Bearer {backend.api_key}"})
                    statuses[name] = "ok" if r.status_code < 400 else f"http_{r.status_code}"
                except Exception as e:
                    statuses[name] = f"error_{type(e).__name__}"
        overall = "ok" if all(v == "ok" for v in statuses.values()) else "degraded"
        return {"status": overall, "backends": statuses}
    # walter
    try:
        async with httpx.AsyncClient(timeout=HEALTH_TIMEOUT) as client:
            r = await client.get(f"{WALTER_BACKEND}/v1/models")
        return {"status": "ok" if r.status_code < 400 else "degraded", "backend": WALTER_BACKEND, "http": r.status_code}
    except Exception as e:
        return {"status": "degraded", "backend": WALTER_BACKEND, "error": type(e).__name__}


# ===========================================================================
# ===  ALFRED MODE  =========================================================
#   routing + model selection + Anthropic translation + tracing
# ===========================================================================
class UpstreamError(Exception):
    def __init__(
        self,
        status: int,
        body: str,
        backend: str,
        *,
        retryable: bool | None = None,
        error_type: str = "upstream_error",
    ):
        self.status = status
        self.body = body
        self.backend = backend
        self.retryable = is_retryable_status(status) if retryable is None else retryable
        self.error_type = error_type
        super().__init__(f"{backend} returned {status}")


def check_auth(authorization: str | None) -> None:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = authorization[7:].strip()
    if not hmac.compare_digest(token, MASTER_KEY):
        raise HTTPException(status_code=401, detail="Invalid bearer token")


def forwarded_headers(request: Request) -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in request.headers.items():
        if k.lower() in HEADER_BLOCKLIST:
            continue
        out[k] = v
    return out


def _external_fallback_allowed(model_name: str, candidate: Backend) -> bool:
    return (
        ALLOW_EXTERNAL_FALLBACK is True
        or model_name in EXTERNAL_FALLBACK_ROUTES
        or candidate.allow_external_fallback is True
    )


def resolve_chain(model_name: str) -> list[Backend]:
    if model_name not in BACKENDS:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model_name}")
    primary = BACKENDS[model_name]
    primary_boundary = _validate_trust_boundary(primary.trust_boundary, primary.base_url)
    chain = [primary]
    for fb in FALLBACKS.get(model_name, []):
        if fb not in BACKENDS:
            continue
        candidate = BACKENDS[fb]
        candidate_boundary = _validate_trust_boundary(candidate.trust_boundary, candidate.base_url)
        if (
            primary_boundary == "local"
            and candidate_boundary != "local"
            and not _external_fallback_allowed(model_name, candidate)
        ):
            log.warning(
                "model=%s fallback=%s blocked: external fallback requires explicit authorization",
                model_name,
                fb,
            )
            continue
        chain.append(candidate)
    return chain


def make_upstream_headers(request: Request, backend: Backend) -> dict[str, str]:
    headers = forwarded_headers(request)
    headers["Authorization"] = f"Bearer {backend.api_key}"
    headers["Content-Type"] = "application/json"
    for k, v in backend.extra_headers.items():
        headers[k] = v
    return headers


async def read_json_body(request: Request) -> Any:
    cl = request.headers.get("content-length")
    if cl and cl.isdigit() and int(cl) > MAX_BODY_BYTES:
        raise HTTPException(status_code=413, detail="request body too large")
    body = await request.body()
    if len(body) > MAX_BODY_BYTES:
        raise HTTPException(status_code=413, detail="request body too large")
    try:
        return json.loads(body)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"invalid JSON: {e}")


async def verify_swap_in_loaded(backend: Backend) -> bool:
    if backend.name not in SWAP_IN_IDENTITY:
        return True
    now = time.monotonic()
    cached = _IDENTITY_CACHE.get(backend.name)
    if cached and (now - cached[0]) < _IDENTITY_CACHE_TTL:
        return cached[1]
    url = f"{backend.base_url.rstrip('/')}/models"
    ok = False
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            r = await client.get(url, headers={"Authorization": f"Bearer {backend.api_key}"})
        if r.status_code < 400:
            data = r.json().get("data") or []
            if data:
                first_id = (data[0].get("id") or "")
                ok = any(s in first_id for s in SWAP_IN_IDENTITY[backend.name])
    except Exception:
        ok = False
    _IDENTITY_CACHE[backend.name] = (now, ok)
    return ok


_SSE_BLOCK_SEP_RE = re.compile(rb"\r\n\r\n|\n\n|\r\n\n|\n\r\n|\r\r(?!\n|$)")
_SSE_BLOCK_SEP_FINAL_RE = re.compile(rb"\r\n\r\n|\n\n|\r\n\n|\n\r\n|\r\r")
_SSE_LINE_SPLIT_RE = re.compile(rb"\r\n|\n|\r")


def parse_sse_blocks(buf: bytes, *, final: bool = False) -> tuple[list[tuple[str, str]], bytes]:
    """Incrementally parse SSE event blocks with full LF, CRLF, and CR support.

    When final=False, a trailing CR-CR sequence at the end of `buf` waits for
    the next chunk so fragmented CRLF-CRLF delimiters are never split early.
    """
    events: list[tuple[str, str]] = []
    sep_re = _SSE_BLOCK_SEP_FINAL_RE if final else _SSE_BLOCK_SEP_RE
    while True:
        match = sep_re.search(buf)
        if match is None:
            break
        block = buf[:match.start()]
        buf = buf[match.end():]
        event_type = "message"
        data_parts: list[str] = []
        for raw_line in _SSE_LINE_SPLIT_RE.split(block):
            if not raw_line or raw_line.startswith(b":"):
                continue
            if raw_line.startswith(b"event:"):
                event_type = raw_line[6:].strip().decode("utf-8", "replace") or "message"
            elif raw_line.startswith(b"data:"):
                val = raw_line[5:]
                if val.startswith(b" "):
                    val = val[1:]
                data_parts.append(val.decode("utf-8", "replace"))
            elif raw_line == b"data":
                data_parts.append("")
        if data_parts:
            events.append((event_type, "\n".join(data_parts)))
    return events, buf


def _has_unparsed_sse_tail(buf: bytes) -> bool:
    """Return True if remaining bytes at EOF contain non-comment, non-blank data."""
    if not buf:
        return False
    for raw_line in _SSE_LINE_SPLIT_RE.split(buf):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith(b":"):
            continue
        return True
    return False


def _extract_sse_error(event_type: str, obj: dict[str, Any]) -> dict[str, Any] | None:
    """Return a normalized error dict if an SSE event represents an upstream error.

    Matches explicit `event: error` frames as well as JSON payloads containing a
    non-null `"error"` field or `type == "error"`.
    """
    etype = (event_type or "").strip().lower()
    raw_err = obj.get("error")
    is_error = etype == "error" or obj.get("type") == "error" or raw_err is not None
    if not is_error:
        return None
    if isinstance(raw_err, dict):
        err_obj = dict(raw_err)
        if not err_obj.get("message"):
            err_obj["message"] = str(obj.get("message") or json.dumps(raw_err))
        if not err_obj.get("type"):
            err_obj["type"] = "upstream_stream_error"
        return err_obj
    if isinstance(raw_err, str) and raw_err.strip():
        return {"type": "upstream_stream_error", "message": raw_err.strip()}
    if raw_err is not None:
        return {"type": "upstream_stream_error", "message": str(raw_err)}
    msg = obj.get("message")
    if isinstance(msg, str) and msg.strip():
        return {"type": str(obj.get("type") or "upstream_stream_error"), "message": msg.strip()}
    return {"type": "upstream_stream_error", "message": json.dumps(obj)}


def _is_valid_openai_sse_chunk(evt: dict[str, Any]) -> bool:
    """Return True if `evt` conforms to an OpenAI streaming completion/usage chunk."""
    return isinstance(evt.get("choices"), list) or isinstance(evt.get("usage"), dict)


def _has_openai_completion_evidence(evt: dict[str, Any]) -> bool:
    """Return True if an OpenAI SSE chunk carries real completion evidence
    (non-empty content/reasoning/refusal delta, tool/function call delta, or
    non-empty finish_reason) rather than metadata/usage-only frames."""
    choices = evt.get("choices")
    if not isinstance(choices, list):
        return False
    for ch in choices:
        if not isinstance(ch, dict):
            continue
        finish_reason = ch.get("finish_reason")
        if isinstance(finish_reason, str) and finish_reason.strip():
            return True
        delta = ch.get("delta")
        if not isinstance(delta, dict):
            continue
        for key in ("content", "reasoning_content", "reasoning", "refusal"):
            val = delta.get(key)
            if isinstance(val, str) and val != "":
                return True
        tool_calls = delta.get("tool_calls")
        if isinstance(tool_calls, list) and any(isinstance(tc, dict) for tc in tool_calls):
            return True
        func_call = delta.get("function_call")
        if isinstance(func_call, dict) and bool(func_call):
            return True
    return False


@dataclass
class NormalizedCompletion:
    id: str
    created: int
    model: str
    content: str | None
    reasoning_content: str | None
    tool_calls: list[dict[str, Any]]
    finish_reason: str
    usage: dict[str, Any] | None


def _normalize_openai_completion(obj: dict[str, Any]) -> NormalizedCompletion:
    """Validate and normalize an OpenAI-style chat completion JSON object into
    one shared representation used by both JSON translation and fake-SSE."""
    if not isinstance(obj, dict):
        raise ValueError("upstream completion must be a JSON object")
    choices = obj.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ValueError("upstream completion missing valid choices[0]")
    choice = choices[0]
    msg = choice.get("message")
    if msg is None:
        msg = {}
    if not isinstance(msg, dict):
        raise ValueError("upstream completion choices[0].message must be an object")

    raw_content = msg.get("content")
    content = raw_content if isinstance(raw_content, str) else None

    raw_reasoning = msg.get("reasoning_content")
    if not (isinstance(raw_reasoning, str) and raw_reasoning):
        raw_reasoning = msg.get("reasoning")
    reasoning_content = raw_reasoning if isinstance(raw_reasoning, str) and raw_reasoning else None

    raw_tool_calls = msg.get("tool_calls")
    tool_calls: list[dict[str, Any]] = []
    if isinstance(raw_tool_calls, list):
        for tc in raw_tool_calls:
            if isinstance(tc, dict):
                tool_calls.append(dict(tc))

    finish_reason = choice.get("finish_reason")
    if not isinstance(finish_reason, str) or not finish_reason:
        finish_reason = "tool_calls" if tool_calls else "stop"

    usage = obj.get("usage") if isinstance(obj.get("usage"), dict) else None
    return NormalizedCompletion(
        id=str(obj.get("id") or f"chatcmpl-{uuid.uuid4().hex[:12]}"),
        created=int(obj.get("created") or int(time.time())),
        model=str(obj.get("model") or ""),
        content=content,
        reasoning_content=reasoning_content,
        tool_calls=tool_calls,
        finish_reason=finish_reason,
        usage=usage,
    )


def _json_to_fake_sse(obj: dict[str, Any]) -> bytes:
    """Adapt a buffered OpenAI completion JSON payload to OpenAI SSE while
    preserving reasoning_content, content, tool_calls, finish_reason, and usage."""
    norm = _normalize_openai_completion(obj)
    delta: dict[str, Any] = {"role": "assistant"}
    if norm.reasoning_content:
        delta["reasoning_content"] = norm.reasoning_content
    if norm.content is not None:
        delta["content"] = norm.content
    elif not norm.reasoning_content and not norm.tool_calls:
        delta["content"] = ""
    if norm.tool_calls:
        delta["tool_calls"] = [
            {"index": i, **tc} if "index" not in tc else dict(tc)
            for i, tc in enumerate(norm.tool_calls)
        ]
    chunk: dict[str, Any] = {
        "id": norm.id,
        "object": "chat.completion.chunk",
        "created": norm.created,
        "model": norm.model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": norm.finish_reason}],
    }
    if norm.usage is not None:
        chunk["usage"] = norm.usage
    return b"data: " + json.dumps(chunk).encode("utf-8") + b"\n\ndata: [DONE]\n\n"


def _validate_nonstream_completion_response(r: Any, backend_name: str) -> dict[str, Any]:
    """Validate media type, JSON syntax, and completion shape inside a backend attempt."""
    headers = getattr(r, "headers", None) or {}
    ctype = (headers.get("content-type") or headers.get("Content-Type") or "").lower()
    if "text/html" in ctype:
        raise UpstreamError(
            502,
            f"unexpected HTML content-type {ctype!r} from {backend_name}",
            backend_name,
            retryable=True,
            error_type="upstream_protocol_error",
        )
    try:
        data = r.json()
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        raise UpstreamError(
            502,
            f"malformed JSON body from {backend_name}: {exc}",
            backend_name,
            retryable=True,
            error_type="upstream_protocol_error",
        ) from exc
    try:
        _normalize_openai_completion(data)
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise UpstreamError(
            502,
            f"invalid completion payload from {backend_name}: {exc}",
            backend_name,
            retryable=True,
            error_type="upstream_protocol_error",
        ) from exc
    return data


def _format_upstream_error_response(err: UpstreamError, *, anthropic: bool) -> JSONResponse:
    parsed: Any = None
    try:
        parsed = json.loads(err.body) if err.body else None
    except json.JSONDecodeError:
        parsed = None
    if anthropic:
        if isinstance(parsed, dict) and parsed.get("type") == "error" and isinstance(parsed.get("error"), dict):
            return JSONResponse(status_code=err.status, content=parsed)
        msg = err.body
        if isinstance(parsed, dict):
            nested = parsed.get("error")
            if isinstance(nested, dict) and nested.get("message"):
                msg = str(nested["message"])
            elif isinstance(parsed.get("detail"), str):
                msg = parsed["detail"]
        return JSONResponse(
            status_code=err.status,
            content={
                "type": "error",
                "error": {
                    "type": "invalid_request_error" if 400 <= err.status < 500 else "api_error",
                    "message": msg or f"upstream {err.backend} returned HTTP {err.status}",
                    "backend": err.backend,
                },
            },
        )
    if isinstance(parsed, dict) and ("error" in parsed or "detail" in parsed):
        return JSONResponse(status_code=err.status, content=parsed)
    return JSONResponse(
        status_code=err.status,
        content={
            "error": {
                "type": err.error_type,
                "message": err.body or f"upstream {err.backend} returned HTTP {err.status}",
                "backend": err.backend,
                "code": err.status,
            }
        },
    )


def _format_gateway_exhausted_response(
    model_name: str,
    last_err: UpstreamError | None,
    attempts: list[dict[str, Any]],
    *,
    anthropic: bool,
) -> JSONResponse:
    detail = f"all backends failed for model={model_name}"
    if last_err:
        detail += f" (last: backend={last_err.backend} status={last_err.status} type={last_err.error_type})"
    if anthropic:
        return JSONResponse(
            status_code=502,
            content={
                "type": "error",
                "error": {
                    "type": "api_error",
                    "code": "all_backends_failed",
                    "message": detail,
                    "attempts": attempts,
                },
            },
        )
    return JSONResponse(
        status_code=502,
        content={
            "error": {
                "type": "upstream_unavailable",
                "code": "all_backends_failed",
                "message": detail,
                "attempts": attempts,
            }
        },
    )


async def _open_upstream_sse_stream(
    backend: Backend,
    body: dict[str, Any],
    headers: dict[str, str],
) -> AsyncIterator[bytes]:
    """Open an upstream streaming chat response and validate at least one
    complete SSE event (or valid buffered JSON completion) BEFORE committing
    HTTP 200 downstream."""
    client: httpx.AsyncClient | None = httpx.AsyncClient(timeout=DEFAULT_TIMEOUT)
    resp: httpx.Response | None = None
    try:
        url = f"{backend.base_url.rstrip('/')}/chat/completions"
        req = client.build_request("POST", url, json=body, headers=headers)
        resp = await client.send(req, stream=True)
        if resp.status_code >= 400:
            text = await resp.aread()
            raise UpstreamError(
                resp.status_code,
                text.decode("utf-8", "replace"),
                backend.name,
                retryable=is_retryable_status(resp.status_code),
            )
        content_type = (resp.headers.get("content-type") or "").lower()
        if "text/event-stream" in content_type:
            # Always consume decoded transport bytes (aiter_bytes) so upstream
            # Content-Encoding (gzip/deflate/br) is transparently decompressed.
            byte_iter = resp.aiter_bytes()
            buf = b""
            prefetched: list[bytes] = []
            found_valid_event = False
            try:
                async for chunk in byte_iter:
                    if not chunk:
                        continue
                    prefetched.append(chunk)
                    buf += chunk
                    events, buf = parse_sse_blocks(buf)
                    for etype, data_str in events:
                        if etype.strip().lower() == "error":
                            found_valid_event = True
                            break
                        if data_str == "[DONE]":
                            raise UpstreamError(
                                502,
                                "upstream SSE stream ended with done marker before any completion event",
                                backend.name,
                                retryable=True,
                                error_type="upstream_protocol_error",
                            )
                        try:
                            evt = json.loads(data_str)
                        except json.JSONDecodeError as exc:
                            raise UpstreamError(
                                502,
                                f"upstream SSE stream sent malformed JSON event: {exc}",
                                backend.name,
                                retryable=True,
                                error_type="upstream_protocol_error",
                            ) from exc
                        if not isinstance(evt, dict):
                            raise UpstreamError(
                                502,
                                "upstream SSE stream event must be a JSON object",
                                backend.name,
                                retryable=True,
                                error_type="upstream_protocol_error",
                            )
                        if _extract_sse_error(etype, evt) is not None:
                            found_valid_event = True
                            break
                        if not _is_valid_openai_sse_chunk(evt):
                            raise UpstreamError(
                                502,
                                "upstream SSE stream event missing expected completion or error schema",
                                backend.name,
                                retryable=True,
                                error_type="upstream_protocol_error",
                            )
                        if _has_openai_completion_evidence(evt):
                            found_valid_event = True
                            break
                    if found_valid_event:
                        break
            except httpx.DecodingError as exc:
                raise UpstreamError(
                    502,
                    f"upstream SSE stream content-encoding decode failed: {exc}",
                    backend.name,
                    retryable=True,
                    error_type="upstream_decoding_error",
                ) from exc
            if not found_valid_event:
                # Check final flush in case stream used CR-CR only without trailing chunk
                final_events, _ = parse_sse_blocks(buf, final=True)
                for etype, data_str in final_events:
                    if etype.strip().lower() == "error":
                        found_valid_event = True
                        break
                    if data_str == "[DONE]":
                        raise UpstreamError(
                            502,
                            "upstream SSE stream ended with done marker before any completion event",
                            backend.name,
                            retryable=True,
                            error_type="upstream_protocol_error",
                        )
                    try:
                        evt = json.loads(data_str)
                    except json.JSONDecodeError:
                        break
                    if not isinstance(evt, dict):
                        break
                    if _extract_sse_error(etype, evt) is not None:
                        found_valid_event = True
                        break
                    if not _is_valid_openai_sse_chunk(evt):
                        break
                    if _has_openai_completion_evidence(evt):
                        found_valid_event = True
                        break
            if not found_valid_event:
                raise UpstreamError(
                    502,
                    "upstream SSE stream ended before delivering any valid completion event",
                    backend.name,
                    retryable=True,
                    error_type="upstream_protocol_error",
                )

            owned_client = client
            owned_resp = resp
            client = None
            resp = None

            async def _stream_rest() -> AsyncIterator[bytes]:
                try:
                    for c in prefetched:
                        yield c
                    async for c in byte_iter:
                        yield c
                finally:
                    await owned_resp.aclose()
                    await owned_client.aclose()

            return _stream_rest()

        body_bytes = await resp.aread()
        if "text/html" in content_type or (content_type and "json" not in content_type and "text/plain" not in content_type):
            raise UpstreamError(
                502,
                f"unexpected content-type {content_type!r} for streaming response",
                backend.name,
                retryable=True,
                error_type="upstream_protocol_error",
            )
        try:
            obj = json.loads(body_bytes)
            fake_sse = _json_to_fake_sse(obj)
        except (json.JSONDecodeError, ValueError, KeyError, IndexError, TypeError) as exc:
            raise UpstreamError(
                502,
                f"malformed buffered JSON response for stream request: {exc}",
                backend.name,
                retryable=True,
                error_type="upstream_protocol_error",
            ) from exc

        async def _one_shot() -> AsyncIterator[bytes]:
            yield fake_sse

        return _one_shot()
    finally:
        if resp is not None:
            await resp.aclose()
        if client is not None:
            await client.aclose()


async def _stream_upstream_chat(backend: Backend, body: dict[str, Any], headers: dict[str, str]) -> AsyncIterator[bytes]:
    stream = await _open_upstream_sse_stream(backend, body, headers)
    async for chunk in stream:
        yield chunk


def _openai_stream_error_sse(message: str) -> bytes:
    payload = {"error": {"type": "upstream_stream_error", "message": message}}
    return f"event: error\ndata: {json.dumps(payload)}\n\n".encode("utf-8")


async def rewrite_stream_model(byte_iter: AsyncIterator[bytes], model_name: str) -> AsyncIterator[bytes]:
    """Rewrite upstream OpenAI SSE chunks to the requested model name with full
    LF/CRLF framing and strict completion/premature-EOF/error integrity.

    Terminal-marker policy: `[DONE]` terminates the stream once real completion
    evidence has been observed; trailing frames/bytes after `[DONE]` are ignored.
    Metadata/usage-only streams without completion evidence never emit `[DONE]`.
    """
    buf = b""
    saw_completion_evidence = False
    saw_finish_reason = False
    saw_done = False
    saw_upstream_error = False
    stream_error: str | None = None
    pending_frames: list[bytes] = []

    def _process_events(events: list[tuple[str, str]]) -> list[bytes]:
        nonlocal saw_completion_evidence, saw_finish_reason, saw_done, saw_upstream_error, stream_error
        out: list[bytes] = []
        for event_type, data_str in events:
            if saw_done or saw_upstream_error or stream_error is not None:
                return out
            if event_type.strip().lower() == "error":
                try:
                    parsed = json.loads(data_str)
                except json.JSONDecodeError:
                    parsed = None
                if isinstance(parsed, dict):
                    err_obj = _extract_sse_error(event_type, parsed)
                    err_payload = parsed if (parsed.get("error") is not None) else {"error": err_obj}
                else:
                    err_payload = {
                        "error": {
                            "type": "upstream_stream_error",
                            "message": data_str or "upstream stream error",
                        }
                    }
                out.append(b"event: error\ndata: " + json.dumps(err_payload).encode("utf-8") + b"\n\n")
                saw_upstream_error = True
                return out
            if data_str == "[DONE]":
                if not saw_completion_evidence:
                    stream_error = "upstream SSE stream sent done marker without completion data"
                    return out
                saw_done = True
                out.append(b"data: [DONE]\n\n")
                return out
            try:
                obj = json.loads(data_str)
            except json.JSONDecodeError as exc:
                stream_error = f"malformed JSON event in upstream SSE stream: {exc}"
                return out
            if not isinstance(obj, dict):
                stream_error = "non-object JSON event in upstream SSE stream"
                return out
            err_obj = _extract_sse_error(event_type, obj)
            if err_obj is not None:
                err_payload = obj if (obj.get("error") is not None) else {"error": err_obj}
                out.append(b"event: error\ndata: " + json.dumps(err_payload).encode("utf-8") + b"\n\n")
                saw_upstream_error = True
                return out
            if not _is_valid_openai_sse_chunk(obj):
                stream_error = "upstream SSE stream sent unrecognized event schema"
                return out
            has_evidence = _has_openai_completion_evidence(obj)
            for ch in obj.get("choices") or []:
                if isinstance(ch, dict):
                    finish = ch.get("finish_reason")
                    if isinstance(finish, str) and finish.strip():
                        saw_finish_reason = True
            if "model" in obj:
                obj["model"] = model_name
            payload = json.dumps(obj).encode("utf-8")
            prefix = f"event: {event_type}\n".encode("utf-8") if event_type != "message" else b""
            frame = prefix + b"data: " + payload + b"\n\n"
            if saw_completion_evidence:
                out.append(frame)
            elif has_evidence:
                saw_completion_evidence = True
                if pending_frames:
                    out.extend(pending_frames)
                    pending_frames.clear()
                out.append(frame)
            else:
                pending_frames.append(frame)
        return out

    try:
        try:
            async for chunk in byte_iter:
                if not chunk:
                    continue
                buf += chunk
                events, buf = parse_sse_blocks(buf)
                for frame in _process_events(events):
                    yield frame
                if stream_error is not None or saw_upstream_error or saw_done:
                    break
        except Exception as exc:
            stream_error = f"upstream stream interrupted ({type(exc).__name__}): {exc}"

        if stream_error is None and not saw_upstream_error and not saw_done:
            final_events, buf = parse_sse_blocks(buf, final=True)
            for frame in _process_events(final_events):
                yield frame

        if stream_error is None and not saw_upstream_error and not saw_done:
            if _has_unparsed_sse_tail(buf):
                stream_error = "upstream SSE stream ended with unterminated trailing data"
            elif not saw_completion_evidence:
                stream_error = "upstream SSE stream ended without completion data"
            elif not saw_finish_reason:
                stream_error = "upstream SSE stream ended prematurely without terminal finish or done marker"

        if stream_error is not None and not saw_upstream_error:
            yield _openai_stream_error_sse(stream_error)
    finally:
        aclose = getattr(byte_iter, "aclose", None)
        if callable(aclose):
            try:
                await aclose()
            except Exception:
                pass


def _alfred_models_list():
    now = int(time.time())
    return {
        "object": "list",
        "data": [
            {"id": name, "object": "model", "created": now, "owned_by": "gordon-gateway"}
            for name in BACKENDS
        ],
    }


@app.get("/v1/models")
async def list_models(authorization: str | None = Header(default=None)):
    if MODE != "alfred":
        return await _walter_passthrough_get()
    check_auth(authorization)
    return _alfred_models_list()


# --- alfred: trace wrapper around a stream so streaming spans close with usage ---
async def _alfred_traced_stream(inner: AsyncIterator[bytes], span, model: str, start: float,
                                user_id: str, user_class: str, anthropic: bool) -> AsyncIterator[bytes]:
    """Wrap a translated SSE stream; tee out usage/tool-call counts for the
    span + metrics without altering the bytes the client receives."""
    input_tokens = 0
    output_tokens = 0
    tool_call_count = 0
    saw_error = False
    buf = b""

    def _inspect_events(events: list[tuple[str, str]]) -> None:
        nonlocal input_tokens, output_tokens, tool_call_count, saw_error
        for etype, data_str in events:
            if etype.strip().lower() == "error":
                saw_error = True
                continue
            if data_str == "[DONE]":
                continue
            try:
                evt = json.loads(data_str)
            except json.JSONDecodeError:
                continue
            if not isinstance(evt, dict):
                continue
            if anthropic:
                if evt.get("type") == "error" or evt.get("error") is not None:
                    saw_error = True
                    continue
                # Anthropic message_delta carries usage; tool_use blocks count.
                if evt.get("type") == "content_block_start" and \
                        evt.get("content_block", {}).get("type") == "tool_use":
                    tool_call_count += 1
                usage = evt.get("usage") or evt.get("message", {}).get("usage")
                if usage:
                    input_tokens = usage.get("input_tokens", input_tokens)
                    output_tokens = usage.get("output_tokens", output_tokens)
            else:
                if evt.get("type") == "error" or evt.get("error") is not None:
                    saw_error = True
                    continue
                tool_call_count += _count_tool_calls_in_chunk(evt)
                inp, out = _extract_usage(evt)
                if inp > 0:
                    input_tokens = inp
                if out > 0:
                    output_tokens = out

    try:
        async for chunk in inner:
            if TRACING_ENABLED:
                buf += chunk
                events, buf = parse_sse_blocks(buf)
                _inspect_events(events)
            yield chunk
        if TRACING_ENABLED and buf:
            final_events, buf = parse_sse_blocks(buf, final=True)
            _inspect_events(final_events)
    except Exception:
        saw_error = True
        raise
    finally:
        if TRACING_ENABLED:
            duration = time.monotonic() - start
            span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
            span.set_attribute("gen_ai.usage.output_tokens", output_tokens)
            span.set_attribute("gen_ai.usage.total_tokens", input_tokens + output_tokens)
            span.set_attribute("gen_ai.response.tool_calls", tool_call_count)
            span.set_attribute("llm.request.duration_s", round(duration, 3))
            if saw_error:
                span.set_attribute("error", True)
            span.end()
            endpoint = "/v1/messages" if anthropic else "/v1/chat/completions"
            _record_metrics(input_tokens, output_tokens, tool_call_count, duration, model, endpoint, user_id, user_class)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request, authorization: str | None = Header(default=None)):
    if MODE != "alfred":
        return await walter_handle(request)

    check_auth(authorization)
    raw_body = await request.body()
    if len(raw_body) > MAX_BODY_BYTES:
        raise HTTPException(status_code=413, detail="request body too large")
    try:
        body = json.loads(raw_body)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"invalid JSON: {e}")
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="body must be a JSON object")
    requested_model = body.get("model")
    if not requested_model:
        raise HTTPException(status_code=400, detail="model is required")
    chain = resolve_chain(requested_model)
    stream = bool(body.get("stream"))
    rid = uuid.uuid4().hex[:8]
    headers_in = dict(request.headers)
    user_id, user_class = _resolve_user(headers_in)
    start = time.monotonic()

    last_err: UpstreamError | None = None
    attempts: list[dict[str, Any]] = []
    for backend in chain:
        if not await verify_swap_in_loaded(backend):
            last_err = UpstreamError(
                503,
                f"{backend.name} not currently loaded on upstream",
                backend.name,
                retryable=True,
                error_type="identity_check_failed",
            )
            attempts.append({"backend": backend.name, "status": 503, "error_type": "identity_check_failed"})
            log.warning("rid=%s backend=%s identity_check_failed", rid, backend.name)
            continue
        upstream_body = dict(body)
        upstream_body["model"] = backend.upstream_model
        # Apply default max_tokens only when omitted/None so callers without an
        # explicit budget do not hit tiny backend defaults, while preserving any
        # explicit caller budget verbatim.
        upstream_body["max_tokens"] = _resolve_max_tokens(upstream_body.get("max_tokens"))
        # o11y: on streaming requests, inject stream_options.include_usage so the
        # upstream emits a final usage chunk and the trace layer records real
        # token counts.
        if stream:
            upstream_body["stream_options"] = {
                **(upstream_body.get("stream_options") or {}),
                "include_usage": True,
            }
        headers = make_upstream_headers(request, backend)
        url = f"{backend.base_url.rstrip('/')}/chat/completions"
        log.info("rid=%s model=%s -> backend=%s stream=%s", rid, requested_model, backend.name, stream)
        try:
            if stream:
                upstream_iter = await _open_upstream_sse_stream(backend, upstream_body, headers)
                span = _start_chat_span(requested_model, True, raw_body, user_id, user_class, current=False)
                translated = rewrite_stream_model(upstream_iter, requested_model)
                return StreamingResponse(
                    _alfred_traced_stream(translated, span, requested_model, start, user_id, user_class, anthropic=False),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
                )
            else:
                span = _start_chat_span(requested_model, False, raw_body, user_id, user_class, current=True)
                try:
                    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
                        r = await client.post(url, json=upstream_body, headers=headers)
                    if r.status_code >= 400:
                        raise UpstreamError(
                            r.status_code,
                            r.text,
                            backend.name,
                            retryable=is_retryable_status(r.status_code),
                        )
                    data = _validate_nonstream_completion_response(r, backend.name)
                    if "model" in data:
                        data["model"] = requested_model
                    duration = time.monotonic() - start
                    inp, out = _extract_usage(data)
                    tcc = _count_tool_calls_in_chunk(data)
                    if TRACING_ENABLED:
                        span.set_attribute("gen_ai.usage.input_tokens", inp)
                        span.set_attribute("gen_ai.usage.output_tokens", out)
                        span.set_attribute("gen_ai.usage.total_tokens", inp + out)
                        span.set_attribute("gen_ai.response.tool_calls", tcc)
                        span.set_attribute("llm.request.duration_s", round(duration, 3))
                    _record_metrics(inp, out, tcc, duration, requested_model, "/v1/chat/completions", user_id, user_class)
                    return JSONResponse(content=data, status_code=r.status_code)
                except Exception:
                    if TRACING_ENABLED:
                        span.set_attribute("error", True)
                    raise
                finally:
                    if TRACING_ENABLED:
                        span.end()
        except UpstreamError as e:
            attempts.append({"backend": e.backend, "status": e.status, "error_type": e.error_type})
            if not e.retryable:
                log.warning("rid=%s backend=%s terminal_status=%s", rid, e.backend, e.status)
                return _format_upstream_error_response(e, anthropic=False)
            last_err = e
            log.warning("rid=%s backend=%s retryable_status=%s type=%s", rid, e.backend, e.status, e.error_type)
            continue
        except (httpx.TransportError, httpx.DecodingError) as e:
            status = 502 if isinstance(e, httpx.DecodingError) else 503
            last_err = UpstreamError(status, str(e), backend.name, retryable=True, error_type=type(e).__name__)
            attempts.append({"backend": backend.name, "status": status, "error_type": type(e).__name__})
            log.warning("rid=%s backend=%s transport_err=%s", rid, backend.name, type(e).__name__)
            continue

    return _format_gateway_exhausted_response(requested_model, last_err, attempts, anthropic=False)


# --- Anthropic Messages translation (alfred) ---
def _content_blocks_to_text(blocks: Any) -> str:
    if isinstance(blocks, str):
        return blocks
    if not isinstance(blocks, list):
        return ""
    parts: list[str] = []
    for b in blocks:
        if isinstance(b, dict) and b.get("type") == "text":
            parts.append(b.get("text", ""))
    return "".join(parts)


def anthropic_to_openai_request(req: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"model": req["model"]}
    msgs: list[dict[str, Any]] = []
    system_msg = req.get("system")
    if system_msg:
        msgs.append({"role": "system", "content": _content_blocks_to_text(system_msg)})
    for m in req.get("messages", []):
        role = m.get("role")
        content = m.get("content")
        if role == "user":
            user_text_parts: list[str] = []
            user_content_blocks: list[dict[str, Any]] = []
            has_multimodal = False
            tool_results: list[dict[str, Any]] = []
            if isinstance(content, str):
                user_text_parts.append(content)
            elif isinstance(content, list):
                for b in content:
                    if not isinstance(b, dict):
                        continue
                    t = b.get("type")
                    if t == "text":
                        txt = b.get("text", "")
                        user_text_parts.append(txt)
                        user_content_blocks.append({"type": "text", "text": txt})
                    elif t == "tool_result":
                        tool_results.append(b)
                    elif t == "image":
                        source = b.get("source")
                        if not isinstance(source, dict):
                            raise ValueError("Anthropic image block requires a 'source' object")
                        src_type = source.get("type")
                        if src_type == "base64":
                            media_type = source.get("media_type")
                            img_data = source.get("data")
                            if not isinstance(media_type, str) or not media_type.strip() or not isinstance(img_data, str) or not img_data:
                                raise ValueError("Anthropic base64 image source requires non-empty 'media_type' and 'data'")
                            image_url = f"data:{media_type.strip()};base64,{img_data}"
                        elif src_type == "url":
                            url_val = source.get("url")
                            if not isinstance(url_val, str) or not url_val.strip():
                                raise ValueError("Anthropic url image source requires non-empty 'url'")
                            image_url = url_val.strip()
                        else:
                            raise ValueError(f"Unsupported Anthropic image source type: {src_type!r}")
                        has_multimodal = True
                        user_content_blocks.append({"type": "image_url", "image_url": {"url": image_url}})
                    else:
                        raise ValueError(f"Unsupported Anthropic user content block type: {t!r}")
            for tr in tool_results:
                tr_content = tr.get("content")
                if isinstance(tr_content, list):
                    tr_text = _content_blocks_to_text(tr_content)
                elif isinstance(tr_content, str):
                    tr_text = tr_content
                else:
                    tr_text = json.dumps(tr_content)
                msgs.append({"role": "tool", "tool_call_id": tr.get("tool_use_id", ""), "content": tr_text})
            if has_multimodal:
                msgs.append({"role": "user", "content": user_content_blocks})
            elif user_text_parts:
                msgs.append({"role": "user", "content": "".join(user_text_parts)})
        elif role == "assistant":
            asst_text_parts: list[str] = []
            tool_calls: list[dict[str, Any]] = []
            if isinstance(content, str):
                asst_text_parts.append(content)
            elif isinstance(content, list):
                for b in content:
                    if not isinstance(b, dict):
                        continue
                    t = b.get("type")
                    if t == "text":
                        asst_text_parts.append(b.get("text", ""))
                    elif t == "thinking":
                        # Assistant thinking blocks are non-functional history in OpenAI format; ignore cleanly.
                        continue
                    elif t == "tool_use":
                        tool_calls.append({
                            "id": b.get("id", ""),
                            "type": "function",
                            "function": {"name": b.get("name", ""), "arguments": json.dumps(b.get("input", {}))},
                        })
                    else:
                        raise ValueError(f"Unsupported Anthropic assistant content block type: {t!r}")
            asst_msg: dict[str, Any] = {"role": "assistant", "content": "".join(asst_text_parts) if asst_text_parts else None}
            if tool_calls:
                asst_msg["tool_calls"] = tool_calls
            msgs.append(asst_msg)
    out["messages"] = msgs
    if "tools" in req and isinstance(req["tools"], list):
        out["tools"] = [
            {"type": "function", "function": {
                "name": t["name"], "description": t.get("description", ""),
                "parameters": t.get("input_schema", {})}}
            for t in req["tools"] if isinstance(t, dict) and "name" in t
        ]
    if "tool_choice" in req and req["tool_choice"] is not None:
        tc = req["tool_choice"]
        if isinstance(tc, dict):
            tc_type = tc.get("type")
            if tc_type == "auto":
                out["tool_choice"] = "auto"
            elif tc_type == "any":
                out["tool_choice"] = "required"
            elif tc_type == "none":
                out["tool_choice"] = "none"
            elif tc_type == "tool":
                tool_name = tc.get("name")
                if not isinstance(tool_name, str) or not tool_name.strip():
                    raise ValueError("Anthropic tool_choice type 'tool' requires a non-empty 'name'")
                out["tool_choice"] = {"type": "function", "function": {"name": tool_name}}
            else:
                raise ValueError(f"Unsupported Anthropic tool_choice type: {tc_type!r}")
            if tc.get("disable_parallel_tool_use") is True:
                out["parallel_tool_calls"] = False
        else:
            raise ValueError(f"Unsupported Anthropic tool_choice value: {tc!r}")
    # Apply default max_tokens only when omitted/None; preserve any explicit
    # caller budget (including small values) verbatim.
    out["max_tokens"] = _resolve_max_tokens(req.get("max_tokens"))
    if "temperature" in req:
        out["temperature"] = req["temperature"]
    if "top_p" in req:
        out["top_p"] = req["top_p"]
    if "stop_sequences" in req:
        out["stop"] = req["stop_sequences"]
    if req.get("stream"):
        out["stream"] = True
        out["stream_options"] = {"include_usage": True}
    # Thinking control: explicit Anthropic `thinking` parameter takes precedence;
    # otherwise opt in per-message by starting a line with "think:" (case-insensitive).
    _think_on = False
    if "thinking" in req and req["thinking"] is not None:
        thinking_cfg = req["thinking"]
        if isinstance(thinking_cfg, dict) and thinking_cfg.get("type") == "enabled":
            _think_on = True
        elif isinstance(thinking_cfg, dict) and thinking_cfg.get("type") == "disabled":
            _think_on = False
        else:
            raise ValueError(f"Unsupported Anthropic thinking configuration: {thinking_cfg!r}")
    else:
        for _m in reversed(out["messages"]):
            if _m.get("role") == "user":
                mc = _m.get("content")
                if isinstance(mc, str):
                    if re.search(r"(?im)^\s*think:\s*", mc):
                        _think_on = True
                        _m["content"] = re.sub(r"(?im)^\s*think:\s*", "", mc, count=1)
                elif isinstance(mc, list):
                    for part in mc:
                        if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str):
                            if re.search(r"(?im)^\s*think:\s*", part["text"]):
                                _think_on = True
                                part["text"] = re.sub(r"(?im)^\s*think:\s*", "", part["text"], count=1)
                                break
                break
    out["enable_thinking"] = _think_on
    out["chat_template_kwargs"] = {**(out.get("chat_template_kwargs") or {}), "enable_thinking": _think_on, "thinking_mode": ("enabled" if _think_on else "disabled")}
    return out


_FINISH_TO_STOP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "content_filter": "end_turn",
    "function_call": "tool_use",
}


def openai_to_anthropic_response(resp: dict[str, Any], model_name: str) -> dict[str, Any]:
    msg_id = "msg_" + uuid.uuid4().hex[:24]
    norm = _normalize_openai_completion(resp)
    content_blocks: list[dict[str, Any]] = []
    if norm.reasoning_content:
        content_blocks.append({"type": "thinking", "thinking": norm.reasoning_content})
    if norm.content:
        content_blocks.append({"type": "text", "text": norm.content})
    for tc in norm.tool_calls:
        fn = tc.get("function", {}) or {}
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError:
            args = {"_raw": fn.get("arguments", "")}
        content_blocks.append({
            "type": "tool_use",
            "id": tc.get("id", f"toolu_{uuid.uuid4().hex[:24]}"),
            "name": fn.get("name", ""),
            "input": args,
        })
    if not content_blocks:
        content_blocks.append({"type": "text", "text": ""})
    stop_reason = _FINISH_TO_STOP.get(norm.finish_reason, "end_turn")
    usage = norm.usage or {}
    return {
        "id": msg_id, "type": "message", "role": "assistant",
        "content": content_blocks, "model": model_name,
        "stop_reason": stop_reason, "stop_sequence": None,
        "usage": {"input_tokens": usage.get("prompt_tokens", 0), "output_tokens": usage.get("completion_tokens", 0)},
    }


def _sse(event: str, data: dict[str, Any]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode("utf-8")


async def openai_stream_to_anthropic(byte_iter: AsyncIterator[bytes], model_name: str) -> AsyncIterator[bytes]:
    """Translate an OpenAI SSE byte stream to Anthropic Messages SSE with full
    LF/CRLF framing and distinct success vs error finalization.

    Terminal-marker policy: `[DONE]` terminates the stream once real completion
    evidence has been observed; trailing frames/bytes after `[DONE]` are ignored.
    Metadata/usage-only streams without completion evidence never emit
    `message_start`, `message_delta`, or `message_stop`.
    """
    msg_id = "msg_" + uuid.uuid4().hex[:24]
    started = False
    saw_completion_evidence = False
    saw_finish_reason = False
    saw_done = False
    upstream_error_obj: dict[str, Any] | None = None
    stream_error: str | None = None
    text_block_open = False
    text_block_idx = -1
    reasoning_block_open = False
    reasoning_block_idx = -1
    next_block_idx = 0
    tool_blocks: dict[Any, dict[str, Any]] = {}
    stop_reason = "end_turn"
    input_tokens = 0
    output_tokens = 0
    buf = b""

    def _process_events(events: list[tuple[str, str]]) -> list[bytes]:
        nonlocal started, saw_completion_evidence, saw_finish_reason, saw_done, upstream_error_obj, stream_error
        nonlocal text_block_open, text_block_idx, reasoning_block_open, reasoning_block_idx
        nonlocal next_block_idx, stop_reason, input_tokens, output_tokens
        out: list[bytes] = []
        for event_type, data_str in events:
            if saw_done or upstream_error_obj is not None or stream_error is not None:
                return out
            if event_type.strip().lower() == "error":
                try:
                    parsed = json.loads(data_str)
                except json.JSONDecodeError:
                    parsed = None
                if isinstance(parsed, dict):
                    upstream_error_obj = _extract_sse_error(event_type, parsed)
                else:
                    upstream_error_obj = {
                        "type": "api_error",
                        "message": data_str or "upstream stream error",
                    }
                return out
            if data_str == "[DONE]":
                if not saw_completion_evidence:
                    stream_error = "upstream SSE stream sent done marker without completion data"
                    return out
                saw_done = True
                return out
            try:
                evt = json.loads(data_str)
            except json.JSONDecodeError as exc:
                stream_error = f"malformed JSON event in upstream SSE stream: {exc}"
                return out
            if not isinstance(evt, dict):
                stream_error = "non-object JSON event in upstream SSE stream"
                return out
            err_obj = _extract_sse_error(event_type, evt)
            if err_obj is not None:
                upstream_error_obj = err_obj
                return out
            if not _is_valid_openai_sse_chunk(evt):
                stream_error = "upstream SSE stream sent unrecognized event schema"
                return out
            usage_chunk = evt.get("usage")
            if isinstance(usage_chunk, dict):
                input_tokens = usage_chunk.get("prompt_tokens", input_tokens)
                output_tokens = usage_chunk.get("completion_tokens", output_tokens)
            if _has_openai_completion_evidence(evt):
                saw_completion_evidence = True
            if saw_completion_evidence and not started:
                started = True
                out.append(_sse("message_start", {"type": "message_start", "message": {
                    "id": msg_id, "type": "message", "role": "assistant", "content": [],
                    "model": model_name, "stop_reason": None, "stop_sequence": None,
                    "usage": {"input_tokens": 0, "output_tokens": 0}}}))
            choices = evt.get("choices") or []
            if not choices:
                continue
            ch = choices[0]
            if not isinstance(ch, dict):
                continue
            delta = ch.get("delta") or {}
            finish = ch.get("finish_reason")
            reasoning_delta = delta.get("reasoning_content")
            if not (isinstance(reasoning_delta, str) and reasoning_delta):
                reasoning_delta = delta.get("reasoning")
            if isinstance(reasoning_delta, str) and reasoning_delta:
                if not reasoning_block_open:
                    reasoning_block_idx = next_block_idx
                    next_block_idx += 1
                    out.append(_sse("content_block_start", {"type": "content_block_start", "index": reasoning_block_idx,
                                                            "content_block": {"type": "thinking", "thinking": ""}}))
                    reasoning_block_open = True
                out.append(_sse("content_block_delta", {"type": "content_block_delta", "index": reasoning_block_idx,
                                                        "delta": {"type": "thinking_delta", "thinking": reasoning_delta}}))
            content = delta.get("content")
            if isinstance(content, str) and content:
                if reasoning_block_open:
                    out.append(_sse("content_block_stop", {"type": "content_block_stop", "index": reasoning_block_idx}))
                    reasoning_block_open = False
                if not text_block_open:
                    text_block_idx = next_block_idx
                    next_block_idx += 1
                    out.append(_sse("content_block_start", {"type": "content_block_start", "index": text_block_idx,
                                                            "content_block": {"type": "text", "text": ""}}))
                    text_block_open = True
                out.append(_sse("content_block_delta", {"type": "content_block_delta", "index": text_block_idx,
                                                        "delta": {"type": "text_delta", "text": content}}))
            for tc in delta.get("tool_calls") or []:
                if not isinstance(tc, dict):
                    continue
                tc_idx_raw = tc.get("index")
                if tc_idx_raw is not None:
                    tc_key: Any = tc_idx_raw
                elif tc.get("id"):
                    tc_key = tc["id"]
                else:
                    tc_key = f"_synth_{len(tool_blocks)}"
                if tc_key not in tool_blocks:
                    if reasoning_block_open:
                        out.append(_sse("content_block_stop", {"type": "content_block_stop", "index": reasoning_block_idx}))
                        reasoning_block_open = False
                    if text_block_open:
                        out.append(_sse("content_block_stop", {"type": "content_block_stop", "index": text_block_idx}))
                        text_block_open = False
                    block_idx = next_block_idx
                    next_block_idx += 1
                    tc_id = tc.get("id") or f"toolu_{uuid.uuid4().hex[:24]}"
                    tc_name = (tc.get("function") or {}).get("name", "") or ""
                    tool_blocks[tc_key] = {"block_idx": block_idx, "id": tc_id, "name": tc_name, "args": ""}
                    out.append(_sse("content_block_start", {"type": "content_block_start", "index": block_idx,
                                    "content_block": {"type": "tool_use", "id": tc_id, "name": tc_name, "input": {}}}))
                fn = tc.get("function") or {}
                if fn.get("name") and not tool_blocks[tc_key]["name"]:
                    tool_blocks[tc_key]["name"] = fn["name"]
                if fn.get("arguments"):
                    tool_blocks[tc_key]["args"] += fn["arguments"]
                    out.append(_sse("content_block_delta", {"type": "content_block_delta",
                                    "index": tool_blocks[tc_key]["block_idx"],
                                    "delta": {"type": "input_json_delta", "partial_json": fn["arguments"]}}))
            if isinstance(finish, str) and finish.strip():
                saw_finish_reason = True
                stop_reason = _FINISH_TO_STOP.get(finish, "end_turn")
        return out

    try:
        try:
            async for chunk in byte_iter:
                if not chunk:
                    continue
                buf += chunk
                events, buf = parse_sse_blocks(buf)
                for frame in _process_events(events):
                    yield frame
                if stream_error is not None or upstream_error_obj is not None or saw_done:
                    break
        except Exception as exc:
            stream_error = f"upstream stream interrupted ({type(exc).__name__}): {exc}"

        if stream_error is None and upstream_error_obj is None and not saw_done:
            final_events, buf = parse_sse_blocks(buf, final=True)
            for frame in _process_events(final_events):
                yield frame

        if stream_error is None and upstream_error_obj is None and not saw_done:
            if _has_unparsed_sse_tail(buf):
                stream_error = "upstream SSE stream ended with unterminated trailing data"
            elif not saw_completion_evidence:
                stream_error = "upstream SSE stream ended without completion data"
            elif not saw_finish_reason:
                stream_error = "upstream SSE stream ended prematurely without finish marker"

        if upstream_error_obj is not None:
            # Distinct error finalization preserving upstream error: NEVER emit message_delta/end_turn or message_stop.
            yield _sse("error", {"type": "error", "error": upstream_error_obj})
            return

        if stream_error is not None:
            # Distinct error finalization: NEVER emit message_delta/end_turn or message_stop.
            yield _sse("error", {"type": "error", "error": {"type": "api_error", "message": stream_error}})
            return

        if reasoning_block_open:
            yield _sse("content_block_stop", {"type": "content_block_stop", "index": reasoning_block_idx})
        if text_block_open:
            yield _sse("content_block_stop", {"type": "content_block_stop", "index": text_block_idx})
        for tb in tool_blocks.values():
            yield _sse("content_block_stop", {"type": "content_block_stop", "index": tb["block_idx"]})
        yield _sse("message_delta", {"type": "message_delta",
                   "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                   "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens}})
        yield _sse("message_stop", {"type": "message_stop"})
    finally:
        aclose = getattr(byte_iter, "aclose", None)
        if callable(aclose):
            try:
                await aclose()
            except Exception:
                pass


@app.post("/v1/messages")
async def messages(request: Request, authorization: str | None = Header(default=None),
                   x_api_key: str | None = Header(default=None)):
    if MODE != "alfred":
        raise HTTPException(status_code=404, detail="/v1/messages not served in walter mode")
    if authorization is None and x_api_key:
        authorization = f"Bearer {x_api_key}"
    check_auth(authorization)
    raw_body = await request.body()
    if len(raw_body) > MAX_BODY_BYTES:
        raise HTTPException(status_code=413, detail="request body too large")
    try:
        a_req = json.loads(raw_body)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"invalid JSON: {e}")
    if not isinstance(a_req, dict):
        raise HTTPException(status_code=400, detail="body must be a JSON object")
    requested_model = a_req.get("model")
    if not requested_model:
        raise HTTPException(status_code=400, detail="model is required")
    stream = bool(a_req.get("stream"))
    rid = uuid.uuid4().hex[:8]
    try:
        o_req = anthropic_to_openai_request(a_req)
    except ValueError as e:
        return JSONResponse(
            status_code=400,
            content={"type": "error", "error": {"type": "invalid_request_error", "message": str(e)}},
        )
    chain = resolve_chain(requested_model)
    headers_in = dict(request.headers)
    user_id, user_class = _resolve_user(headers_in)
    start = time.monotonic()
    # span prompt summary uses the OpenAI-shaped body so it stays uniform.
    span_body = json.dumps(o_req).encode("utf-8")

    last_err: UpstreamError | None = None
    attempts: list[dict[str, Any]] = []
    for backend in chain:
        if not await verify_swap_in_loaded(backend):
            last_err = UpstreamError(
                503,
                f"{backend.name} not currently loaded on upstream",
                backend.name,
                retryable=True,
                error_type="identity_check_failed",
            )
            attempts.append({"backend": backend.name, "status": 503, "error_type": "identity_check_failed"})
            log.warning("rid=%s backend=%s identity_check_failed", rid, backend.name)
            continue
        body = dict(o_req)
        body["model"] = backend.upstream_model
        headers = make_upstream_headers(request, backend)
        log.info("rid=%s anthropic model=%s -> backend=%s stream=%s", rid, requested_model, backend.name, stream)
        try:
            if stream:
                upstream_iter = await _open_upstream_sse_stream(backend, body, headers)
                span = _start_chat_span(requested_model, True, span_body, user_id, user_class, current=False)
                translated = openai_stream_to_anthropic(upstream_iter, requested_model)
                return StreamingResponse(
                    _alfred_traced_stream(translated, span, requested_model, start, user_id, user_class, anthropic=True),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
                )
            else:
                span = _start_chat_span(requested_model, False, span_body, user_id, user_class, current=True)
                try:
                    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
                        url = f"{backend.base_url.rstrip('/')}/chat/completions"
                        r = await client.post(url, json=body, headers=headers)
                    if r.status_code >= 400:
                        raise UpstreamError(
                            r.status_code,
                            r.text,
                            backend.name,
                            retryable=is_retryable_status(r.status_code),
                        )
                    oai = _validate_nonstream_completion_response(r, backend.name)
                    a_resp = openai_to_anthropic_response(oai, requested_model)
                    duration = time.monotonic() - start
                    inp, out = _extract_usage(oai)
                    tcc = _count_tool_calls_in_chunk(oai)
                    if TRACING_ENABLED:
                        span.set_attribute("gen_ai.usage.input_tokens", inp)
                        span.set_attribute("gen_ai.usage.output_tokens", out)
                        span.set_attribute("gen_ai.usage.total_tokens", inp + out)
                        span.set_attribute("gen_ai.response.tool_calls", tcc)
                        span.set_attribute("llm.request.duration_s", round(duration, 3))
                    _record_metrics(inp, out, tcc, duration, requested_model, "/v1/messages", user_id, user_class)
                    return JSONResponse(content=a_resp)
                except Exception:
                    if TRACING_ENABLED:
                        span.set_attribute("error", True)
                    raise
                finally:
                    if TRACING_ENABLED:
                        span.end()
        except UpstreamError as e:
            attempts.append({"backend": e.backend, "status": e.status, "error_type": e.error_type})
            if not e.retryable:
                log.warning("rid=%s backend=%s terminal_status=%s", rid, e.backend, e.status)
                return _format_upstream_error_response(e, anthropic=True)
            last_err = e
            log.warning("rid=%s backend=%s retryable_status=%s type=%s", rid, e.backend, e.status, e.error_type)
            continue
        except (httpx.TransportError, httpx.DecodingError) as e:
            status = 502 if isinstance(e, httpx.DecodingError) else 503
            last_err = UpstreamError(status, str(e), backend.name, retryable=True, error_type=type(e).__name__)
            attempts.append({"backend": backend.name, "status": status, "error_type": type(e).__name__})
            log.warning("rid=%s backend=%s transport_err=%s", rid, backend.name, type(e).__name__)
            continue

    return _format_gateway_exhausted_response(requested_model, last_err, attempts, anthropic=True)


@app.post("/v1/messages/count_tokens")
async def count_tokens(request: Request, authorization: str | None = Header(default=None),
                       x_api_key: str | None = Header(default=None)):
    if MODE != "alfred":
        raise HTTPException(status_code=404, detail="not served in walter mode")
    if authorization is None and x_api_key:
        authorization = f"Bearer {x_api_key}"
    check_auth(authorization)
    a_req = await read_json_body(request)
    if not isinstance(a_req, dict):
        raise HTTPException(status_code=400, detail="body must be a JSON object")
    try:
        o_req = anthropic_to_openai_request(a_req)
    except ValueError as e:
        return JSONResponse(
            status_code=400,
            content={"type": "error", "error": {"type": "invalid_request_error", "message": str(e)}},
        )
    payload = json.dumps({"messages": o_req.get("messages", []), "tools": o_req.get("tools", [])})
    estimate = max(1, len(payload) // 4)
    return {"input_tokens": estimate}


# ===========================================================================
# ===  WALTER MODE  =========================================================
#   blockops tool-call translation + loop guards + ctx truncation + memory
#   warnings + KV-cache purge, with OTLP tracing layered on.
#   Logic ported verbatim from local blockops-proxy.py; HTTP handling
#   re-implemented on httpx/FastAPI.
# ===========================================================================
QWEN_TOOL_SYSTEM = """\
# Tools

You may call one or more functions to assist with the user query.

Available functions:

{tool_list}

When you need to call a function, output EXACTLY this format (one per line):
TOOL_CALL: {{"name": "<function-name>", "arguments": <args-json-object>}}

Always include a visible text response in addition to any tool calls."""

_TOOL_PROMPT_TARGET_CHARS = 10240

if psutil is not None:
    TOTAL_RAM_BYTES = psutil.virtual_memory().total
else:
    TOTAL_RAM_BYTES = 64 * 1024 ** 3
TOTAL_RAM_GB = TOTAL_RAM_BYTES / (1024 ** 3)

MEM_WARN_TIERS = [
    (0.05, "Crash likely imminent — save work NOW",
           "🚨 *Memory CRITICAL* — only {avail:.1f} GB free ({pct:.0f}% of {total:.0f} GB). Crash likely imminent. Save work now."),
    (0.10, "High pressure — /compact or /new soon",
           "🔴 *Memory HIGH* — {avail:.1f} GB free ({pct:.0f}% of {total:.0f} GB). High pressure — consider /compact or /new soon."),
    (0.15, "Session growing large",
           "⚠️ *Memory WARNING* — {avail:.1f} GB free ({pct:.0f}% of {total:.0f} GB). Session growing large."),
    (0.20, "Session is getting long",
           "⚡ *Memory NOTICE* — {avail:.1f} GB free ({pct:.0f}% of {total:.0f} GB). Session is getting long."),
]

# Walter conversation state. Resource-wide state (_last_request_time,
# _inflight_per_backend and _tool_prompt_cache) remains process-wide by design;
# everything that describes one conversation lives in the bounded LRU below.
@dataclass
class WalterSessionState:
    last_actual_prompt_tokens: int = 0
    banner_sent: bool = False
    last_message_count: int = 0
    session_nudge_count: int = 0
    parse_error_counts: OrderedDict[str, int] = field(default_factory=OrderedDict)
    last_prompt_tokens: int = 0
    is_new_session: bool = True
    last_seen: float = field(default_factory=time.monotonic)


_SESSION_ID_HEADERS = (
    "x-claude-code-session-id",
    "x-gordon-session-id",
    "x-hermes-session-id",
    "x-session-id",
    "x-conversation-id",
)
_SESSION_METADATA_KEYS = ("session_id", "conversation_id", "thread_id")
_session_states: OrderedDict[str, WalterSessionState] = OrderedDict()
_last_request_time = 0.0
_inflight_per_backend: dict = {}
_tool_prompt_cache: dict = {}


def _stable_json(value: Any) -> str:
    """Canonical, non-secret-bearing input for a deterministic fingerprint."""
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return repr(value)


def _session_key(headers: dict, data: dict, user_id: str) -> tuple[str, str]:
    """Return an opaque, stable conversation key and its non-sensitive source.

    Claude Code's explicit session header is authoritative. Agent IDs scope
    subagents within the parent session because their message histories and loop
    counters are independent. Other clients can send X-Gordon-Session-Id (or a
    conventional equivalent) or metadata.session_id. The final fallback hashes
    the immutable conversation root so ordinary stateless OpenAI clients still
    retain state as their message list grows.
    """
    normalized_headers = {str(k).lower(): str(v) for k, v in headers.items()}
    explicit = ""
    source = ""
    for header in _SESSION_ID_HEADERS:
        value = normalized_headers.get(header, "").strip()
        if value:
            explicit = value
            source = header
            break

    if not explicit:
        metadata = data.get("metadata")
        if isinstance(metadata, dict):
            for key in _SESSION_METADATA_KEYS:
                value = metadata.get(key)
                if value is not None and str(value).strip():
                    explicit = str(value).strip()
                    source = f"metadata.{key}"
                    break

    agent_id = normalized_headers.get("x-claude-code-agent-id", "").strip()
    if explicit:
        raw_identity = _stable_json({"user": user_id, "session": explicit, "agent": agent_id})
        digest = hashlib.sha256(raw_identity.encode("utf-8", "replace")).hexdigest()[:32]
        return f"explicit:{digest}", source

    messages = data.get("messages")
    if not isinstance(messages, list):
        messages = []
    root_messages = []
    first_user_seen = False
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role == "system" and not first_user_seen:
            root_messages.append({"role": role, "content": message.get("content")})
        elif role == "user":
            root_messages.append({"role": role, "content": message.get("content")})
            first_user_seen = True
            break
    fallback_identity = _stable_json({
        "user": user_id,
        "agent": agent_id,
        "root": root_messages,
    })
    digest = hashlib.sha256(fallback_identity.encode("utf-8", "replace")).hexdigest()[:32]
    return f"fallback:{digest}", "conversation-root"


def _prune_session_states(now: float | None = None) -> None:
    """Expire idle entries and enforce the configured LRU ceiling."""
    if now is None:
        now = time.monotonic()
    cutoff = now - SESSION_STATE_TTL_SECONDS
    while _session_states:
        _key, oldest = next(iter(_session_states.items()))
        if oldest.last_seen >= cutoff:
            break
        _session_states.popitem(last=False)
    while len(_session_states) > SESSION_STATE_MAX:
        _session_states.popitem(last=False)


def _get_session_state(session_key: str, now: float | None = None) -> WalterSessionState:
    """Fetch/create one session state while maintaining bounded LRU order."""
    if now is None:
        now = time.monotonic()
    _prune_session_states(now)
    state = _session_states.pop(session_key, None)
    if state is None:
        while len(_session_states) >= SESSION_STATE_MAX:
            _session_states.popitem(last=False)
        state = WalterSessionState(last_seen=now)
    else:
        state.last_seen = now
    _session_states[session_key] = state
    return state


def _record_parse_error(counts: OrderedDict[str, int], signature: str) -> None:
    """Increment a malformed-call signature without unbounded per-session keys."""
    if signature in counts:
        counts[signature] += 1
        counts.move_to_end(signature)
        return
    while len(counts) >= SESSION_PARSE_ERROR_KEYS_MAX:
        counts.popitem(last=False)
    counts[signature] = 1


def _rewrite_backend_path(path_qs: str) -> str:
    if not BACKEND_STRIP_PATH_PREFIX:
        return path_qs
    if path_qs.startswith(BACKEND_STRIP_PATH_PREFIX + "/"):
        return path_qs[len(BACKEND_STRIP_PATH_PREFIX):]
    if path_qs == BACKEND_STRIP_PATH_PREFIX:
        return "/"
    return path_qs


def _get_inflight(backend_url: str) -> int:
    return _inflight_per_backend.get(backend_url, 0)


def _inc_inflight(backend_url: str) -> int:
    _inflight_per_backend[backend_url] = _inflight_per_backend.get(backend_url, 0) + 1
    return _inflight_per_backend[backend_url]


def _dec_inflight(backend_url: str) -> int:
    _inflight_per_backend[backend_url] = max(0, _inflight_per_backend.get(backend_url, 0) - 1)
    return _inflight_per_backend[backend_url]


def _backend_for_model(model_name: str) -> str:
    if model_name in MODEL_BACKENDS:
        return MODEL_BACKENDS[model_name]
    if model_name in MODEL_ALIASES:
        return MODEL_BACKENDS.get(MODEL_ALIASES[model_name], WALTER_BACKEND)
    return WALTER_BACKEND


async def purge_backend_cache():
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.delete(f"{WALTER_BACKEND}/v1/cache?type=all")
            bolog(f"CACHE_PURGE status={resp.status_code} body={resp.content[:200]!r}")
    except Exception as e:
        bolog(f"CACHE_PURGE_ERROR {e}")


async def get_cache_size_mb() -> float:
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{WALTER_BACKEND}/v1/cache/stats")
            if resp.status_code != 200:
                return -1
            data = resp.json()
            for key in ("prefix_cache", "paged_prefix", "memory_aware_prefix"):
                entry = data.get(key)
                if entry and isinstance(entry, dict):
                    size_bytes = entry.get("size_bytes") or entry.get("bytes") or 0
                    if size_bytes:
                        return size_bytes / (1024 * 1024)
            return 0.0
    except Exception as e:
        bolog(f"CACHE_STATS_ERROR {e}")
        return -1


def truncate_messages(messages: list, target_tokens: int) -> list:
    if not messages:
        return messages

    def est(msgs):
        return int(sum(len(m.get("content", "") or "") for m in msgs) / 2.5)

    system_msgs = []
    conv_msgs = []
    for m in messages:
        if m.get("role") == "system" and not conv_msgs:
            system_msgs.append(m)
        else:
            conv_msgs.append(m)
    system_tokens = est(system_msgs)
    budget = target_tokens - system_tokens
    if budget <= 0:
        return system_msgs + conv_msgs[-2:]
    head_msgs = []
    head_tokens = 0
    for i, m in enumerate(conv_msgs):
        if m.get("role") == "user":
            head_msgs.append(m)
            head_tokens += int(len(m.get("content", "") or "") / 2.5)
            if i + 1 < len(conv_msgs) and conv_msgs[i + 1].get("role") == "assistant":
                head_msgs.append(conv_msgs[i + 1])
                head_tokens += int(len(conv_msgs[i + 1].get("content", "") or "") / 2.5)
            break
    head_budget = int(budget * 0.15)
    if head_tokens > head_budget:
        head_msgs = head_msgs[:1]
        head_tokens = int(len(head_msgs[0].get("content", "") or "") / 2.5) if head_msgs else 0
        if head_tokens > head_budget:
            head_msgs = []
            head_tokens = 0
    budget -= head_tokens
    head_ids = {id(m) for m in head_msgs}
    kept = []
    used = 0
    for m in reversed(conv_msgs):
        if id(m) in head_ids:
            continue
        content = m.get("content", "") or ""
        msg_tokens = int(len(content) / 2.5)
        remaining = budget - used
        if msg_tokens > remaining:
            if kept:
                break
            if remaining > 100 and isinstance(content, str):
                chars_budget = int(remaining * 2.5)
                head = chars_budget // 2
                tail = chars_budget - head - 100
                truncated = (content[:head] + f"\n\n[...{len(content) - head - tail} chars truncated...]\n\n" + content[-tail:])
                m = dict(m)
                m["content"] = truncated
                kept.append(m)
                used += int(len(truncated) / 2.5)
            break
        kept.append(m)
        used += msg_tokens
    kept.reverse()
    while kept and kept[0].get("role") != "user":
        kept.pop(0)
    dropped = len(conv_msgs) - len(head_msgs) - len(kept)
    if dropped > 0:
        notice_text = f"\n\n[Context truncated: {dropped} middle messages removed to fit memory limits. Original task context and recent conversation preserved.]"
        if system_msgs:
            system_msgs = [dict(system_msgs[0])]
            system_msgs[0]["content"] = (system_msgs[0].get("content") or "") + notice_text
        else:
            system_msgs = [{"role": "system", "content": notice_text.strip()}]
    return system_msgs + head_msgs + kept


def make_session_start_notice(avail_gb: float) -> str:
    lines = ["⚙️ *[TESTING MODE] Compaction is disabled.*",
             f"This session will grow until memory runs out. "
             f"Start a new session (/new) before you hit the limit.\n",
             f"Memory warnings will appear at (machine total: {TOTAL_RAM_GB:.0f} GB):"]
    for frac, label, msg_template in reversed(MEM_WARN_TIERS):
        threshold_gb = TOTAL_RAM_GB * frac
        emoji = msg_template.split()[0]
        lines.append(f"  {emoji} < {frac*100:.0f}% free  (< {threshold_gb:.1f} GB)  — {label}")
    lines.append(f"\nCurrent: {avail_gb:.1f} GB free of {TOTAL_RAM_GB:.0f} GB "
                 f"({avail_gb/TOTAL_RAM_GB*100:.0f}% free)\n"
                 "─────────────────────────────────")
    return "\n".join(lines) + "\n\n"


def get_memory_warning() -> tuple[str | None, float]:
    if psutil is None:
        return None, 0.0
    try:
        mem = psutil.virtual_memory()
        available_gb = mem.available / (1024 ** 3)
        available_frac = mem.available / TOTAL_RAM_BYTES
        used_pct = mem.percent
        bolog(f"MEM_CHECK available={available_gb:.1f}GB ({available_frac*100:.0f}% free) "
              f"used={used_pct:.0f}% total={TOTAL_RAM_GB:.0f}GB")
        for threshold_frac, _label, msg_template in MEM_WARN_TIERS:
            if available_frac < threshold_frac:
                msg = msg_template.format(avail=available_gb, total=TOTAL_RAM_GB, pct=available_frac*100)
                bolog(f"MEM_WARN tier=<{threshold_frac*100:.0f}% available={available_gb:.1f}GB")
                return msg, available_gb
        return None, available_gb
    except Exception as e:
        bolog(f"MEM_CHECK_ERROR {e}")
        return None, 0.0


def estimate_actual_tokens(messages: list, session_state: WalterSessionState | None = None) -> int:
    total_chars = sum(len(m.get("content", "") or "") for m in messages)
    char_estimate = int(total_chars / 3.0)
    last_actual = session_state.last_actual_prompt_tokens if session_state is not None else 0
    return max(char_estimate, last_actual)


def estimate_cache_tokens(prompt_tokens: int, session_state: WalterSessionState) -> tuple[int, int]:
    if session_state.is_new_session:
        cache_write = prompt_tokens
        cache_read = 0
        session_state.is_new_session = False
    else:
        cache_read = prompt_tokens
        cache_write = 0
    session_state.last_prompt_tokens = prompt_tokens
    return cache_read, cache_write


def _compact_params(params: dict) -> str:
    if not params or params.get("type") != "object":
        return "{}"
    props = params.get("properties", {})
    required = set(params.get("required", []))
    if not props:
        return "{}"
    parts = []
    for name, schema in props.items():
        typ = schema.get("type", "any")
        if "enum" in schema:
            typ = "|".join(str(v) for v in schema["enum"])
        desc = schema.get("description", "")
        req_mark = "" if name in required else "?"
        entry = f"{name}{req_mark}: {typ}"
        if desc:
            short_desc = desc[:60].rstrip()
            if len(desc) > 60:
                short_desc += "…"
            entry += f' "{short_desc}"'
        parts.append(entry)
    return "{" + ", ".join(parts) + "}"


def build_tool_system_prompt(tools: list) -> str:
    seen_names = set()
    sorted_tools = []
    for t in sorted((x for x in tools if x.get("type") == "function"),
                    key=lambda x: x.get("function", {}).get("name", "")):
        name = t.get("function", {}).get("name", "")
        if name and name not in seen_names:
            seen_names.add(name)
            sorted_tools.append(t)
    tools = sorted_tools
    cache_key = frozenset(seen_names)
    if cache_key in _tool_prompt_cache:
        return _tool_prompt_cache[cache_key]
    tool_defs = []
    for t in tools:
        if t.get("type") == "function":
            fn = t["function"]
            name = fn.get("name", "?")
            desc = fn.get("description", "")
            params = fn.get("parameters", {})
            compact = _compact_params(params)
            if desc:
                first_dot = desc.find(". ")
                if 0 < first_dot < 120:
                    desc = desc[:first_dot + 1]
                elif len(desc) > 120:
                    desc = desc[:120].rstrip() + "…"
            tool_defs.append(f"{name}({compact}): {desc}")
    result = QWEN_TOOL_SYSTEM.format(tool_list="\n".join(tool_defs))
    if cache_key not in _tool_prompt_cache:
        sizes = [(fn.get("name", "?"), len(defn)) for fn, defn in
                 zip((t.get("function", {}) for t in tools), tool_defs)]
        biggest = sorted(sizes, key=lambda x: -x[1])[:5]
        bolog(f"TOOL_SIZES total={sum(s for _, s in sizes)}chars count={len(sizes)} "
              f"biggest={[(n, s) for n, s in biggest]}")
    current_len = len(result)
    if current_len < _TOOL_PROMPT_TARGET_CHARS:
        padding_needed = _TOOL_PROMPT_TARGET_CHARS - current_len
        filler_block = ("x" * 64) + "\n"
        num_blocks = padding_needed // len(filler_block)
        remainder = padding_needed % len(filler_block)
        padding = (filler_block * num_blocks) + ("x" * max(0, remainder - 1)) + "\n"
        result = result + "\n\n<!-- padding for stable cache prefix -->\n" + padding
    elif current_len > _TOOL_PROMPT_TARGET_CHARS:
        bolog(f"TOOL_PROMPT_OVERSIZE actual={current_len} target={_TOOL_PROMPT_TARGET_CHARS} — cache stability reduced")
    _tool_prompt_cache[cache_key] = result
    return result


def sanitize_messages_for_backend(messages: list) -> list:
    out = []
    for m in messages:
        m = m.copy()
        role = m.get("role", "")
        if role == "assistant" and m.get("tool_calls"):
            parts = []
            if m.get("content"):
                parts.append(m["content"])
            for tc in m["tool_calls"]:
                fn = tc.get("function", {})
                name = fn.get("name", "unknown")
                args = fn.get("arguments", "{}")
                parts.append(f'TOOL_CALL: {{"name": "{name}", "arguments": {args}}}')
            m["content"] = "\n".join(parts)
            m.pop("tool_calls", None)
        if role == "tool":
            tool_id = m.get("tool_call_id", "?")
            content = m.get('content', '') or ''
            if isinstance(content, str) and len(content) > TOOL_RESULT_COMPRESSION_THRESHOLD:
                keep_head = TOOL_RESULT_COMPRESSION_THRESHOLD // 2
                keep_tail = TOOL_RESULT_COMPRESSION_THRESHOLD - keep_head - 100
                elided = len(content) - keep_head - keep_tail
                content = (content[:keep_head] + f"\n\n[...{elided} chars elided for context efficiency...]\n\n" + content[-keep_tail:])
            m["role"] = "user"
            m["content"] = f"[Tool result for {tool_id}]\n{content}"
            m.pop("tool_call_id", None)
            m.pop("name", None)
        if m.get("content") is None:
            m["content"] = ""
        out.append(m)
    return out


def inject_tool_prompt(messages: list, tool_system: str) -> list:
    messages = [m.copy() for m in messages]
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            content = messages[i].get("content") or ""
            if isinstance(content, str):
                messages[i]["content"] = tool_system + "\n\n---\n\n" + content
            else:
                messages[i]["content"] = ([{"type": "text", "text": tool_system + "\n\n---\n\n"}]
                                          + (content if isinstance(content, list) else []))
            return messages
    messages.insert(0, {"role": "system", "content": tool_system})
    return messages


def inject_system_notice(messages: list, notice: str) -> list:
    messages = [m.copy() for m in messages]
    for m in messages:
        if m.get("role") == "system":
            m["content"] = (m.get("content") or "") + "\n\n" + notice
            return messages
    messages.insert(0, {"role": "system", "content": notice})
    return messages


def strip_thinking(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"^.*?</think>", "", text, flags=re.DOTALL).strip()
    if text.startswith("Thinking Process:"):
        text = re.sub(r"^Thinking Process:[\s\S]*?(?=\n(?!\s*[\*\-]|\d+[\.\)])(?=[A-Z])|\Z)", "", text).strip()
    while re.match(r"^\d+[\.\)]\s*\**\s*(Analyze|Determine|Draft|Refin|Final|Formul|Construct)", text):
        text = re.sub(r"(?:^\d+[\.\)]\s.*(?:\n(?!\d+[\.\)]|[A-Z][a-z]).*)*)(?:\n\n|\Z)", "", text, flags=re.MULTILINE).strip()
    return text


def parse_tool_calls(content: str, parse_error_counts: OrderedDict[str, int] | None = None):
    tool_calls = []
    json_tc_pattern = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
    for raw in json_tc_pattern.findall(content):
        try:
            parsed = json.loads(raw)
            tool_calls.append({"id": f"call_{uuid.uuid4().hex[:8]}", "type": "function",
                               "function": {"name": parsed.get("name", ""), "arguments": json.dumps(parsed.get("arguments", {}))}})
        except json.JSONDecodeError:
            bolog(f"TOOL_PARSE_ERROR json_tc raw={raw[:100]}")
    tc_line_pattern = re.compile(r"TOOL_CALL:[ \t]*")
    tc_spans = []
    _decoder = json.JSONDecoder(strict=False)
    for m in tc_line_pattern.finditer(content):
        json_start = m.end()
        try:
            parsed, end_idx = _decoder.raw_decode(content, json_start)
        except (json.JSONDecodeError, ValueError) as e:
            parsed = None
            end_idx = json_start
            tail = content[json_start:]
            tail_rstripped = tail.rstrip()
            bare_name_m = re.match(r'\{\s*"([A-Za-z_][A-Za-z0-9_-]*)"\s*,', tail_rstripped)
            if bare_name_m:
                bare_name = bare_name_m.group(1)
                rewritten = '{"name": "' + bare_name + '"' + tail_rstripped[bare_name_m.end() - 1:]
                try:
                    parsed_try, end_try = _decoder.raw_decode(rewritten, 0)
                    if isinstance(parsed_try, dict):
                        parsed = parsed_try
                        delta = len(rewritten) - len(tail_rstripped)
                        end_idx = json_start + (end_try - delta)
                        bolog(f"TOOL_PARSE_AUTOCORRECTED bare-name→name at {json_start} name={bare_name!r}")
                except (json.JSONDecodeError, ValueError):
                    pass
            if parsed is None:
                missing_args_m = re.match(r'(\{\s*"name"\s*:\s*"[^"]+"\s*,\s*)\{', tail_rstripped)
                if missing_args_m:
                    inject_at = missing_args_m.end() - 1
                    rewritten = (tail_rstripped[:inject_at] + '"arguments": ' + tail_rstripped[inject_at:])
                    for extra in ("", "}", "}}", "}}}"):
                        try:
                            parsed_try, end_try = _decoder.raw_decode(rewritten + extra, 0)
                            if isinstance(parsed_try, dict):
                                parsed = parsed_try
                                delta = len(rewritten) - len(tail_rstripped)
                                end_idx = json_start + (end_try - delta - len(extra))
                                bolog(f"TOOL_PARSE_AUTOCORRECTED missing-arguments-key at {json_start} appended={len(extra)}")
                                break
                        except (json.JSONDecodeError, ValueError):
                            continue
            if parsed is None:
                candidates = []
                next_marker = re.search(r"\n[ \t]*(TOOL_CALL:|<tool_call>)", tail)
                if next_marker:
                    candidates.append(tail[:next_marker.start()])
                for nm in re.finditer(r"\n", tail):
                    candidates.append(tail[:nm.start()])
                candidates.append(tail)
                candidates = sorted(set(candidates), key=len)
                for cand in candidates:
                    cand_rstripped = cand.rstrip()
                    if not cand_rstripped.startswith("{"):
                        continue
                    for extra in ("", "}", "}}", "}}}"):
                        try:
                            parsed_try, end_try = _decoder.raw_decode(cand_rstripped + extra, 0)
                            if isinstance(parsed_try, dict):
                                parsed = parsed_try
                                end_idx = json_start + len(cand_rstripped)
                                bolog(f"TOOL_PARSE_RECOVERED bounded={len(cand_rstripped)} appended={len(extra)} at {json_start}")
                                break
                        except (json.JSONDecodeError, ValueError):
                            continue
                    if parsed is not None:
                        break
            if parsed is None:
                raw_sig = content[json_start:json_start+80]
                bolog(f"TOOL_PARSE_ERROR tc_line at {json_start}: {e} raw={raw_sig!r}")
                if parse_error_counts is not None:
                    _record_parse_error(parse_error_counts, raw_sig)
                eol = content.find("\n", json_start)
                if eol == -1:
                    eol = len(content)
                tc_spans.append((m.start(), eol))
                continue
        if not isinstance(parsed, dict):
            bolog(f"TOOL_PARSE_ERROR tc_line non-dict at {json_start}: {type(parsed).__name__}")
            continue
        tc_spans.append((m.start(), end_idx))
        tool_calls.append({"id": f"call_{uuid.uuid4().hex[:8]}", "type": "function",
                           "function": {"name": parsed.get("name", ""), "arguments": json.dumps(parsed.get("arguments", {}))}})
    openclaw_pattern = re.compile(r"<(?:tool_call>\s*)?<function=(\w+)>(.*?)</function>(?:\s*</tool_call>)?", re.DOTALL)
    for func_name, params_block in openclaw_pattern.findall(content):
        arguments = {}
        for param_match in re.finditer(r"<parameter=(\w+)>\s*(.*?)\s*</parameter>", params_block, re.DOTALL):
            arguments[param_match.group(1)] = param_match.group(2)
        tool_calls.append({"id": f"call_{uuid.uuid4().hex[:8]}", "type": "function",
                           "function": {"name": func_name, "arguments": json.dumps(arguments)}})
    fc_pattern = re.compile(r"<function_calls>(.*?)</function_calls>", re.DOTALL)
    invoke_pattern = re.compile(r'<invoke\s+([^>]*)>(.*?)(?:</\s*invoke>|(?<!<)invoke>)', re.DOTALL)
    invoke_attr_pattern = re.compile(r'(\w+)\s*=\s*"([^"]*)"')
    parameter_tag_pattern = re.compile(r'<parameter\s+name="([^"]+)"[^>]*>\s*(.*?)\s*</parameter>', re.DOTALL)

    def _emit_invoke(attrs_str: str, inv_body: str) -> None:
        attrs = dict(invoke_attr_pattern.findall(attrs_str))
        func_name = attrs.pop("name", "")
        if not func_name:
            return
        arguments = {}
        for pm in parameter_tag_pattern.finditer(inv_body):
            arguments[pm.group(1)] = pm.group(2)
        if not arguments:
            param_key = attrs.pop("parameter", None)
            if param_key and inv_body.strip():
                arguments[param_key] = inv_body.strip()
            else:
                for k, v in attrs.items():
                    if k != "tool":
                        arguments[k] = v
                if not arguments and inv_body.strip():
                    arguments["value"] = inv_body.strip()
        tool_calls.append({"id": f"call_{uuid.uuid4().hex[:8]}", "type": "function",
                           "function": {"name": func_name, "arguments": json.dumps(arguments)}})

    consumed_invoke_spans: list[tuple[int, int]] = []
    for fc_match in fc_pattern.finditer(content):
        for inv_match in invoke_pattern.finditer(fc_match.group(1)):
            _emit_invoke(inv_match.group(1), inv_match.group(2))
        consumed_invoke_spans.append((fc_match.start(), fc_match.end()))
    for inv_match in invoke_pattern.finditer(content):
        if any(s <= inv_match.start() < e for s, e in consumed_invoke_spans):
            continue
        _emit_invoke(inv_match.group(1), inv_match.group(2))

    bare_json_blobs: list[str] = []
    if not tool_calls:
        def _balanced_json_blocks(text):
            i, n = 0, len(text)
            while i < n:
                if text[i] != "{":
                    i += 1
                    continue
                depth, in_str, esc = 0, False, False
                start = i
                while i < n:
                    c = text[i]
                    if in_str:
                        if esc:
                            esc = False
                        elif c == "\\":
                            esc = True
                        elif c == '"':
                            in_str = False
                    else:
                        if c == '"':
                            in_str = True
                        elif c == "{":
                            depth += 1
                        elif c == "}":
                            depth -= 1
                            if depth == 0:
                                yield text[start:i + 1]
                                i += 1
                                break
                    i += 1
                else:
                    return
        for blob in _balanced_json_blocks(content):
            try:
                parsed = json.loads(blob)
            except (json.JSONDecodeError, ValueError):
                continue
            if not isinstance(parsed, dict):
                continue
            name = parsed.get("name")
            args = parsed.get("arguments")
            if not isinstance(name, str) or not isinstance(args, dict):
                continue
            tool_calls.append({"id": f"call_{uuid.uuid4().hex[:8]}", "type": "function",
                               "function": {"name": name, "arguments": json.dumps(args)}})
            bare_json_blobs.append(blob)
            bolog(f"TOOL_PARSE_BARE_JSON name={name!r} blob_len={len(blob)}")

    if not tool_calls:
        if tc_spans:
            out = []
            cursor = 0
            for span_start, span_end in sorted(tc_spans):
                out.append(content[cursor:span_start])
                out.append("[tool call dropped — malformed]")
                cursor = span_end
            out.append(content[cursor:])
            return strip_thinking("".join(out)), None
        return strip_thinking(content), None

    cleaned = json_tc_pattern.sub("", content)
    out = []
    cursor = 0
    for m in tc_line_pattern.finditer(cleaned):
        json_start = m.end()
        try:
            _parsed, end_idx = _decoder.raw_decode(cleaned, json_start)
        except (json.JSONDecodeError, ValueError):
            tail = cleaned[json_start:].rstrip()
            bare_m = re.match(r'\{\s*"([A-Za-z_][A-Za-z0-9_-]*)"\s*,', tail)
            if not bare_m:
                continue
            rewritten = '{"name": "' + bare_m.group(1) + '"' + tail[bare_m.end() - 1:]
            try:
                _parsed, end_try = _decoder.raw_decode(rewritten, 0)
                delta = len(rewritten) - len(tail)
                end_idx = json_start + (end_try - delta)
            except (json.JSONDecodeError, ValueError):
                continue
        out.append(cleaned[cursor:m.start()])
        cursor = end_idx
    out.append(cleaned[cursor:])
    cleaned = "".join(out)
    cleaned = openclaw_pattern.sub("", cleaned)
    cleaned = fc_pattern.sub("", cleaned)
    cleaned = invoke_pattern.sub("", cleaned)
    for blob in bare_json_blobs:
        cleaned = cleaned.replace(blob, "", 1)
    cleaned = strip_thinking(cleaned.strip())
    return cleaned, tool_calls


def rewrite_response(resp_data: dict,
                     parse_error_counts: OrderedDict[str, int] | None = None) -> dict:
    try:
        choice = resp_data["choices"][0]
        message = choice.get("message", {})
        content = message.get("content") or ""
        cleaned_content, tool_calls = parse_tool_calls(content, parse_error_counts)
        if tool_calls:
            message["content"] = cleaned_content or None
            message["tool_calls"] = tool_calls
            choice["finish_reason"] = "tool_calls"
            bolog(f"TOOL_CALLS parsed={[tc['function']['name'] for tc in tool_calls]}")
        else:
            message["content"] = cleaned_content
        resp_data["choices"][0]["message"] = message
    except Exception as e:
        bolog(f"REWRITE_ERROR {e}")
    return resp_data


async def fetch_backend_blocking(body_bytes: bytes, headers: dict, path_qs: str, alias: str = None):
    backend = MODEL_BACKENDS.get(alias, WALTER_BACKEND)
    url = backend + _rewrite_backend_path(path_qs)
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(700.0, connect=10.0)) as client:
            resp = await client.post(url, headers=headers, content=body_bytes)
            return resp.content, resp.status_code
    except Exception as e:
        bolog(f"BACKEND_ERROR {e}")
        return json.dumps({"error": str(e)}).encode(), 502


def _assistant_tool_signature(m: dict) -> tuple[str, str] | None:
    if m.get("role") != "assistant":
        return None
    tcs = m.get("tool_calls")
    strict_parts = []
    fuzzy_parts = []
    if tcs:
        for tc in tcs:
            fn = tc.get("function", {}) if isinstance(tc, dict) else {}
            name = fn.get("name", "?")
            raw = fn.get("arguments", "") or ""
            strict_parts.append(f"{name}::{raw}")
            try:
                parsed = json.loads(raw) if raw else {}
                canon = json.dumps(parsed, sort_keys=True, separators=(",", ":"))
                fuzzy_hash = hashlib.sha1(canon.encode("utf-8")).hexdigest()[:12]
            except Exception:
                prefix = re.sub(r"\s+", "", raw)[:80]
                fuzzy_hash = "raw:" + hashlib.sha1(prefix.encode("utf-8", "replace")).hexdigest()[:12]
            fuzzy_parts.append(f"{name}::{fuzzy_hash}")
        return ("|".join(strict_parts), "|".join(fuzzy_parts))
    content = m.get("content") or ""
    if isinstance(content, str) and "TOOL_CALL:" in content:
        tc_lines = [ln.strip() for ln in content.splitlines() if ln.strip().startswith("TOOL_CALL:")]
        if tc_lines:
            joined = "|".join(tc_lines)
            fuzzy_hash = hashlib.sha1(re.sub(r"\s+", "", joined).encode("utf-8", "replace")).hexdigest()[:12]
            return (joined, "inline::" + fuzzy_hash)
    return None


def detect_tool_loop(messages: list, threshold: int, window: int | None = None) -> str | None:
    if window is None:
        window = threshold + 2
    strict_sigs: list[str] = []
    fuzzy_sigs: list[str] = []
    strict_map: dict[str, str] = {}
    for m in reversed(messages):
        if m.get("role") != "assistant":
            continue
        sig = _assistant_tool_signature(m)
        if sig is None:
            if len(fuzzy_sigs) >= threshold:
                break
            return None
        strict, fuzzy = sig
        strict_sigs.append(strict)
        fuzzy_sigs.append(fuzzy)
        strict_map.setdefault(fuzzy, strict)
        if len(fuzzy_sigs) >= window:
            break
    if len(fuzzy_sigs) < threshold:
        return None
    if len(set(strict_sigs[:threshold])) == 1:
        return strict_sigs[0]
    top_fuzzy, cnt = Counter(fuzzy_sigs).most_common(1)[0]
    if cnt >= threshold:
        return strict_map.get(top_fuzzy, top_fuzzy)
    return None


def detect_tool_cycle(messages: list, rep_threshold: int, min_cycle_len: int = 2,
                      max_cycle_len: int = LOOP_DETECT_CYCLE_MAX_LEN) -> tuple[list[str], int, int] | None:
    needed = max_cycle_len * rep_threshold
    strict_sigs: list[str] = []
    fuzzy_sigs: list[str] = []
    strict_map: dict[str, str] = {}
    for m in reversed(messages):
        if m.get("role") != "assistant":
            continue
        sig = _assistant_tool_signature(m)
        if sig is None:
            if len(fuzzy_sigs) >= min_cycle_len * rep_threshold:
                break
            return None
        strict, fuzzy = sig
        strict_sigs.append(strict)
        fuzzy_sigs.append(fuzzy)
        strict_map.setdefault(fuzzy, strict)
        if len(fuzzy_sigs) >= needed:
            break
    fuzzy_sigs.reverse()
    for L in range(min_cycle_len, max_cycle_len + 1):
        if len(fuzzy_sigs) < L * rep_threshold:
            continue
        tail = fuzzy_sigs[-L * rep_threshold:]
        pattern = tail[:L]
        if len(set(pattern)) < 2:
            continue
        if all(tail[r * L:(r + 1) * L] == pattern for r in range(1, rep_threshold)):
            return ([strict_map.get(f, f) for f in pattern], rep_threshold, L)
    return None


def make_sse_chunk(request_id, model, delta, finish_reason=None):
    chunk = {
        "id": request_id, "object": "chat.completion.chunk",
        "created": int(datetime.datetime.now().timestamp()), "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(chunk)}\n\n".encode()


# Walter trace helper: records a span+metrics for a completed walter turn.
def _walter_trace(model: str, body: bytes, in_tok: int, out_tok: int, tcc: int,
                  duration: float, user_id: str, user_class: str, streaming: bool, error: bool = False):
    if not TRACING_ENABLED:
        return
    span = _start_chat_span(model, streaming, body, user_id, user_class, current=False)
    span.set_attribute("gen_ai.usage.input_tokens", in_tok)
    span.set_attribute("gen_ai.usage.output_tokens", out_tok)
    span.set_attribute("gen_ai.usage.total_tokens", in_tok + out_tok)
    span.set_attribute("gen_ai.response.tool_calls", tcc)
    span.set_attribute("llm.request.duration_s", round(duration, 3))
    if error:
        span.set_attribute("error", True)
    span.end()
    _record_metrics(in_tok, out_tok, tcc, duration, model, "/v1/chat/completions", user_id, user_class)


async def _walter_passthrough_get():
    """Used by /v1/models GET in walter mode (passthrough to backend)."""
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
            resp = await client.get(f"{WALTER_BACKEND}/v1/models")
            return Response(content=resp.content, status_code=resp.status_code,
                            media_type=resp.headers.get("content-type", "application/json"))
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=502)


async def walter_handle(request: Request):
    """Walter-mode chat completions handler. Port of blockops-proxy.handle()'s
    chat path. Non-chat paths handled by walter_passthrough()."""
    global _last_request_time
    _last_request_time = time.time()
    start = time.monotonic()
    body_bytes = await request.body()
    headers = {k: v for k, v in request.headers.items()
               if k.lower() not in ("host", "content-length", "transfer-encoding")}
    try:
        data = json.loads(body_bytes)
    except Exception as e:
        bolog(f"BODY_PARSE_ERROR path={request.url.path} err={e}")
        return JSONResponse({"error": "invalid JSON"}, status_code=400)

    request_headers = dict(request.headers)
    user_id, user_class = _resolve_user(request_headers)
    session_key, session_source = _session_key(request_headers, data, user_id)
    session_state = _get_session_state(session_key)
    requested_model = data.get("model", "<none>")
    bolog(f"INBOUND path={request.url.path} body_len={len(body_bytes)} model={requested_model} "
          f"session={session_key} session_source={session_source}")

    messages = data.get("messages", [])
    client_message_count = len(messages)
    tools = data.pop("tools", None)
    tool_choice = data.pop("tool_choice", None)
    if tool_choice is not None:
        if tool_choice == "none" or (isinstance(tool_choice, dict) and tool_choice.get("type") == "none"):
            # Explicit no-tool semantics: suppress tool prompt injection and route
            # through the direct no-tools path.
            tools = None
        elif tool_choice == "auto" or (isinstance(tool_choice, dict) and tool_choice.get("type") == "auto"):
            pass
        else:
            bolog(f"UNSUPPORTED_TOOL_CHOICE value={tool_choice!r}")
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "type": "invalid_request_error",
                        "code": "unsupported_tool_choice",
                        "message": f"Walter prompt-injected tool mode supports tool_choice 'auto' or 'none'; got {tool_choice!r}",
                    }
                },
            )

    requested_backend = _backend_for_model(data.get("model", ""))
    backend_inflight = _get_inflight(requested_backend)
    if backend_inflight >= MAX_CONCURRENT:
        bolog(f"CONCURRENCY_REJECTED backend={requested_backend} inflight={backend_inflight} limit={MAX_CONCURRENT}")
        return JSONResponse(status_code=429, headers={"Retry-After": "30"},
            content={"error": {"message": f"Model busy: {backend_inflight} requests already in progress at {requested_backend} (limit {MAX_CONCURRENT}). Retry in a few seconds.",
                               "type": "rate_limit_exceeded", "code": 429}})

    if len(messages) > MAX_MESSAGES:
        bolog(f"REQUEST_REJECTED messages={len(messages)} exceeds MAX_MESSAGES={MAX_MESSAGES}")
        return JSONResponse(status_code=413,
            content={"error": {"message": f"Too many messages ({len(messages)}). Start a new session (/new).",
                               "type": "request_too_large", "code": 413}})

    # Only accepted requests advance conversation state. A busy/oversized
    # request must not look like a reset to the next valid turn.
    previous_message_count = session_state.last_message_count
    is_session_reset = client_message_count < previous_message_count
    if is_session_reset:
        session_state.banner_sent = False
        session_state.session_nudge_count = 0
        session_state.parse_error_counts.clear()
        session_state.last_actual_prompt_tokens = 0
        session_state.last_prompt_tokens = 0
        session_state.is_new_session = True
        bolog(f"SESSION_RESET session={session_key} msgs={previous_message_count}→{client_message_count} "
              "— session state cleared; shared backend cache retained")
    session_state.last_message_count = client_message_count

    client_wanted_stream = bool(data.get("stream", False))
    worst_sig, worst_count = None, 0
    for sig, cnt in session_state.parse_error_counts.items():
        if cnt > worst_count:
            worst_sig, worst_count = sig, cnt
    if worst_count >= REPEAT_PARSE_ERROR_STOP:
        request_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        model_alias = data.get("model", "")
        bolog(f"PARSE_ERROR_STOP count={worst_count} sig={worst_sig!r}")
        notice = (f"⚠️ The model emitted a malformed tool call {worst_count}× in a row "
                  f"that the parser could not recover. Stopping the turn to avoid leaking raw "
                  f"tool-call JSON. Offending signature: `{worst_sig[:120]}`. Rephrase the request or run `/new`.")
        session_state.parse_error_counts.clear()
        _walter_trace(model_alias, body_bytes, 0, 0, 0, time.monotonic() - start, user_id, user_class, client_wanted_stream)
        if client_wanted_stream:
            return _walter_simple_stream(request_id, model_alias, notice)
        return JSONResponse(content={"id": request_id, "object": "chat.completion", "model": model_alias,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": notice}, "finish_reason": "stop"}]})

    stop_sig = detect_tool_loop(messages, LOOP_DETECT_STOP_THRESHOLD)
    stop_cycle = None if stop_sig else detect_tool_cycle(messages, LOOP_DETECT_CYCLE_STOP_REPS)
    if stop_sig or stop_cycle:
        request_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        model_alias = data.get("model", "local")
        if stop_sig:
            bolog(f"LOOP_DETECTED threshold={LOOP_DETECT_STOP_THRESHOLD} messages={len(messages)} sig={stop_sig[:200]!r}")
            notice = (f"⚠️ Tool-call loop detected: the same tool was invoked {LOOP_DETECT_STOP_THRESHOLD} times in a row "
                      f"with identical arguments, including after a corrective nudge. Stopping the turn. "
                      f"If this was intentional, rephrase the request or run `/new`.")
        else:
            pattern_strict, reps, cycle_len = stop_cycle
            bolog(f"LOOP_CYCLE_DETECTED reps={reps} len={cycle_len} messages={len(messages)} sigs={[s[:120] for s in pattern_strict]!r}")
            preview = "; ".join(f"({i+1}) `{s[:120]}{'...' if len(s) > 120 else ''}`" for i, s in enumerate(pattern_strict))
            notice = (f"⚠️ Tool-call cycle detected: alternated between {cycle_len} tool calls for {reps} full cycles "
                      f"({cycle_len * reps} calls), including after a corrective nudge. Stopping the turn. "
                      f"Offending pattern: {preview}. If this was intentional, rephrase the request or run `/new`.")
        _walter_trace(model_alias, body_bytes, 0, 0, 0, time.monotonic() - start, user_id, user_class, client_wanted_stream)
        if client_wanted_stream:
            return _walter_simple_stream(request_id, model_alias, notice, role_first=True)
        return JSONResponse(content={"id": request_id, "object": "chat.completion", "created": int(time.time()),
            "model": model_alias, "choices": [{"index": 0, "message": {"role": "assistant", "content": notice},
            "finish_reason": "stop"}], "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}})

    nudge_sig = detect_tool_loop(messages, LOOP_DETECT_NUDGE_THRESHOLD)
    nudge_cycle = None if nudge_sig else detect_tool_cycle(messages, LOOP_DETECT_CYCLE_NUDGE_REPS)
    _NUDGE_MARKER = "[loop-guard]"
    last_msg = messages[-1] if messages else {}
    already_nudged = (last_msg.get("role") == "user" and isinstance(last_msg.get("content"), str)
                      and _NUDGE_MARKER in last_msg["content"])
    if (nudge_sig or nudge_cycle) and not already_nudged:
        if nudge_sig:
            bolog(f"LOOP_NUDGE threshold={LOOP_DETECT_NUDGE_THRESHOLD} messages={len(messages)} sig={nudge_sig[:200]!r}")
            nudge_text = (f"{_NUDGE_MARKER} You just issued the same tool call {LOOP_DETECT_NUDGE_THRESHOLD} times in a row "
                          f"with identical arguments. Stop repeating it. Read the prior tool results, identify why the call "
                          f"isn't producing the outcome you expect, and take a different approach (different args, a different "
                          f"tool, or report a blocker to the user). If you cannot make progress, respond in plain text "
                          f"explaining what's blocking you. Do not emit another tool call.")
        else:
            pattern_strict, reps, cycle_len = nudge_cycle
            bolog(f"LOOP_CYCLE_NUDGE reps={reps} len={cycle_len} messages={len(messages)} sigs={[s[:120] for s in pattern_strict]!r}")
            preview = "; ".join(f"({i+1}) `{s[:120]}{'...' if len(s) > 120 else ''}`" for i, s in enumerate(pattern_strict))
            nudge_text = (f"{_NUDGE_MARKER} You are cycling between {cycle_len} tool calls in a repeating pattern. You have now "
                          f"completed {reps} full cycles ({cycle_len * reps} calls) without making progress. The calls are: {preview}. "
                          f"Stop the cycle. Read the prior tool results carefully: if the earliest call in the cycle already "
                          f"succeeded, you are undoing its work by running the next one. If all calls are failing the same way, "
                          f"the approach is wrong, try a different one or report the blocker to the user in plain text. Do not "
                          f"emit another tool call that matches this pattern.")
        session_state.session_nudge_count += 1
        bolog(f"SESSION_NUDGE_COUNT session={session_key} "
              f"count={session_state.session_nudge_count}/{LOOP_DETECT_SESSION_NUDGE_LIMIT}")
        if session_state.session_nudge_count >= LOOP_DETECT_SESSION_NUDGE_LIMIT:
            bolog(f"LOOP_SESSION_STOP session={session_key} count={session_state.session_nudge_count} "
                  f"limit={LOOP_DETECT_SESSION_NUDGE_LIMIT} messages={len(messages)}")
            request_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
            model_alias = data.get("model", "local")
            notice = (f"⚠️ Session-wide loop guard tripped: the model has been nudged {LOOP_DETECT_SESSION_NUDGE_LIMIT} times "
                      f"in this session without making progress (tool-call signatures keep drifting so no single pattern tripped "
                      f"the per-pattern hard stop). Stopping the turn. Run `/new` to reset the session or rephrase the request.")
            _walter_trace(model_alias, body_bytes, 0, 0, 0, time.monotonic() - start, user_id, user_class, client_wanted_stream)
            if client_wanted_stream:
                return _walter_simple_stream(request_id, model_alias, notice, role_first=True)
            return JSONResponse(content={"id": request_id, "object": "chat.completion", "created": int(time.time()),
                "model": model_alias, "choices": [{"index": 0, "message": {"role": "assistant", "content": notice},
                "finish_reason": "stop"}], "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}})
        messages.append({"role": "user", "content": nudge_text})
        data["messages"] = messages

    user_msgs = [m for m in messages[:client_message_count] if m.get("role") == "user"]
    is_session_start = not session_state.banner_sent and len(user_msgs) == 1
    if is_session_start:
        session_state.banner_sent = True
        session_state.is_new_session = True
        _, avail_gb = get_memory_warning()
        notice = make_session_start_notice(avail_gb)
        messages = inject_system_notice(messages, notice)
        bolog(f"SESSION_START session={session_key} — notice injected into system message "
              f"(avail={avail_gb:.1f}GB)")

    if tools:
        return await _walter_tools_path(request, data, headers, messages, tools, requested_backend,
                                        is_session_start, body_bytes, user_id, user_class, start,
                                        session_state)
    return await _walter_no_tools_path(request, data, headers, messages, is_session_start,
                                       body_bytes, user_id, user_class, start, session_state)


def _walter_simple_stream(request_id, model_alias, notice, role_first=False):
    async def gen():
        if role_first:
            yield make_sse_chunk(request_id, model_alias, {"role": "assistant"})
            yield make_sse_chunk(request_id, model_alias, {"content": notice})
        else:
            yield make_sse_chunk(request_id, model_alias, {"role": "assistant", "content": notice})
        yield make_sse_chunk(request_id, model_alias, {}, "stop")
        yield b"data: [DONE]\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"})


async def _walter_tools_path(request, data, headers, messages, tools, requested_backend,
                             is_session_start, body_bytes, user_id, user_class, start,
                             session_state: WalterSessionState):
    tool_system = build_tool_system_prompt(tools)
    messages = inject_tool_prompt(messages, tool_system)
    messages = sanitize_messages_for_backend(messages)
    data["messages"] = messages
    data.pop("tools", None)
    data.pop("tool_choice", None)
    data["max_tokens"] = _resolve_max_tokens(data.get("max_tokens"))
    data["enable_thinking"] = False
    # 2026-06-18 fix: tool turns were sent with no temperature -> M3 defaults to greedy 0.0,
    # which makes it loop its native <invoke> tool tokens into gibberish. Non-greedy floor breaks the loop.
    data.setdefault("temperature", 0.3)
    data.setdefault("top_p", 0.95)
    normalize_upstream_model(data, bolog)

    est_tokens = estimate_actual_tokens(messages, session_state)
    bolog(f"TOOLS_INJECTED count={len(tools)} model_req={data.get('model','?')} est_tokens~{est_tokens}")

    if est_tokens >= TOKEN_HARD_THRESHOLD:
        bolog(f"CTX_LARGE est_tokens={est_tokens} threshold={TOKEN_HARD_THRESHOLD} — truncating")
        messages = truncate_messages(messages, 150_000)
        data["messages"] = messages
        session_state.last_actual_prompt_tokens = 0
        est_tokens = estimate_actual_tokens(messages, session_state)
        bolog(f"CTX_TRUNCATED est_tokens_after={est_tokens} kept_msgs={len(messages)}")

    context_warning = None
    if est_tokens >= TOKEN_WARN_THRESHOLD:
        context_warning = (f"⚠️ *Context warning* — ~{est_tokens // 1000}k tokens "
                           f"(approaching {TOKEN_HARD_THRESHOLD // 1000}k limit). Consider running `/compact` soon.\n\n")
        bolog(f"CTX_SOFT_WARN est_tokens={est_tokens}")

    data["stream"] = False
    out_body = json.dumps(data).encode()
    model_alias = data.get("model", "qwen35")
    request_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"

    backend_count = _inc_inflight(requested_backend)
    bolog(f"INFLIGHT_INC backend={requested_backend} count={backend_count}")

    async def gen():
        in_tok = out_tok = tcc = 0
        try:
            if is_session_start:
                yield make_sse_chunk(request_id, model_alias, {"role": "assistant", "content": f"🧠 *Active model: {_MODEL_DISPLAY}*\n\n"})
            if context_warning:
                yield make_sse_chunk(request_id, model_alias, {"content": context_warning})

            backend_task = asyncio.create_task(fetch_backend_blocking(out_body, headers, request.url.path, alias=data.get('model')))
            while not backend_task.done():
                yield b": keepalive\n\n"
                try:
                    await asyncio.wait_for(asyncio.shield(backend_task), timeout=float(HEARTBEAT_INTERVAL))
                    break
                except asyncio.TimeoutError:
                    continue
            resp_bytes, resp_status = await backend_task
            if resp_status != 200:
                bolog(f"TOOL_BACKEND_NON200 status={resp_status}")
                yield make_sse_chunk(request_id, model_alias, {"content": f"[backend error {resp_status}]"}, "stop")
                yield b"data: [DONE]\n\n"
                _walter_trace(model_alias, body_bytes, 0, 0, 0, time.monotonic() - start, user_id, user_class, True, error=True)
                return

            resp_data = json.loads(resp_bytes)
            usage = resp_data.get('usage', {})
            if usage:
                bolog(f"RESPONSE usage={usage}")
                session_state.last_actual_prompt_tokens = usage.get(
                    "prompt_tokens", usage.get("input_tokens", session_state.last_actual_prompt_tokens),
                )
            resp_str = json.dumps(resp_data)
            for path, alias in ALIAS_REVERSE.items():
                resp_str = resp_str.replace(path, alias)
            resp_data = json.loads(resp_str)
            resp_data = rewrite_response(resp_data, session_state.parse_error_counts)

            choice = resp_data.get("choices", [{}])[0]
            message = choice.get("message", {})
            content = message.get("content") or ""
            tool_calls = message.get("tool_calls")
            if not content.strip() and not tool_calls:
                bolog("EMPTY_CONTENT_FALLBACK")
                content = "I'm here — could you say that again?"

            if tool_calls:
                tcc = len(tool_calls)
                if content:
                    yield make_sse_chunk(request_id, model_alias, {"role": "assistant", "content": content})
                for i, tc in enumerate(tool_calls):
                    hdr_delta = {"tool_calls": [{"index": i, "id": tc["id"], "type": "function",
                                                 "function": {"name": tc["function"]["name"], "arguments": ""}}]}
                    if i == 0:
                        hdr_delta["role"] = "assistant"
                    yield make_sse_chunk(request_id, model_alias, hdr_delta)
                    yield make_sse_chunk(request_id, model_alias, {"tool_calls": [{"index": i, "function": {"arguments": tc["function"]["arguments"]}}]})
                yield make_sse_chunk(request_id, model_alias, {}, "tool_calls")
                bolog(f"TOOL_CALLS parsed={[tc['function']['name'] for tc in tool_calls]}")
            else:
                yield make_sse_chunk(request_id, model_alias, {"role": "assistant", "content": content})
                yield make_sse_chunk(request_id, model_alias, {}, "stop")

            if not usage:
                est_prompt = estimate_actual_tokens(messages, session_state)
                est_completion = len(content) // 4
                usage = {"prompt_tokens": est_prompt, "completion_tokens": est_completion, "total_tokens": est_prompt + est_completion}
            in_tok = usage.get("prompt_tokens", 0)
            out_tok = usage.get("completion_tokens", 0)
            usage_chunk = make_sse_chunk(request_id, model_alias, {}, None)
            usage_data = json.loads(usage_chunk[6:-2])
            usage_data["usage"] = {"prompt_tokens": in_tok, "completion_tokens": out_tok, "total_tokens": in_tok + out_tok}
            yield f"data: {json.dumps(usage_data)}\n\n".encode()
            yield b"data: [DONE]\n\n"
        except Exception as e:
            bolog(f"TOOL_STREAM_ERROR {e}")
        finally:
            c = _dec_inflight(requested_backend)
            bolog(f"INFLIGHT_DEC backend={requested_backend} count={c}")
            _walter_trace(model_alias, body_bytes, in_tok, out_tok, tcc, time.monotonic() - start, user_id, user_class, True)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"})


async def _walter_no_tools_path(request, data, headers, messages, is_session_start,
                                body_bytes, user_id, user_class, start,
                                session_state: WalterSessionState):
    original_model = data.get("model", "<none>")
    # Capture the client's streaming intent BEFORE upstream-body mutation so we
    # can turn on usage injection for the trace layer (o11y: complete token
    # usage on streaming requests).
    client_wants_stream = bool(data.get("stream"))
    normalize_upstream_model(data, bolog)
    data["enable_thinking"] = False
    if client_wants_stream:
        # so a streaming upstream emits a final usage chunk -> the span records
        # real input/output tokens instead of 0/0.
        data["stream_options"] = {**(data.get("stream_options") or {}), "include_usage": True}

    est_tokens = estimate_actual_tokens(messages, session_state)
    if est_tokens >= TOKEN_HARD_THRESHOLD:
        bolog(f"CTX_LARGE est_tokens={est_tokens} threshold={TOKEN_HARD_THRESHOLD} — truncating")
        messages = truncate_messages(messages, 8_000)
        data["messages"] = messages
        est_tokens = estimate_actual_tokens(messages, session_state)
        bolog(f"CTX_TRUNCATED est_tokens_after={est_tokens}")

    ctx_warn_text = None
    if est_tokens >= TOKEN_WARN_THRESHOLD:
        ctx_warn_text = (f"⚠️ *Context warning* — ~{est_tokens // 1000}k tokens "
                         f"(approaching {TOKEN_HARD_THRESHOLD // 1000}k limit). Consider running `/compact` soon.\n\n")

    data["messages"] = sanitize_messages_for_backend(data["messages"])
    out_body = json.dumps(data).encode()
    backend = _backend_for_model(original_model)
    url = backend + _rewrite_backend_path(request.url.path)
    backend_count = _inc_inflight(backend)
    bolog(f"INFLIGHT_INC backend={backend} count={backend_count}")
    in_tok = out_tok = 0
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=10.0)) as client:
            resp = await client.post(url, headers=headers, content=out_body)
            resp_bytes = resp.content
            try:
                resp_data = json.loads(resp_bytes)
                usage = resp_data.get("usage", {})
                if usage:
                    bolog(f"RESPONSE usage={usage}")
                    session_state.last_actual_prompt_tokens = usage.get(
                        "prompt_tokens", usage.get("input_tokens", session_state.last_actual_prompt_tokens),
                    )
                else:
                    est_prompt = estimate_actual_tokens(messages, session_state)
                    est_completion = len(str(resp_data.get("choices", [{}])[0].get("message", {}).get("content", ""))) // 4
                    est_cache_read, est_cache_write = estimate_cache_tokens(est_prompt, session_state)
                    resp_data["usage"] = {"prompt_tokens": est_prompt, "completion_tokens": est_completion,
                                          "total_tokens": est_prompt + est_completion,
                                          "cache_read_tokens": est_cache_read, "cache_write_tokens": est_cache_write}
                    usage = resp_data["usage"]
                for choice in resp_data.get("choices", []):
                    msg = choice.get("message", {})
                    if msg.get("content"):
                        msg["content"] = strip_thinking(msg["content"])
                first_msg = resp_data.get("choices", [{}])[0].get("message", {})
                orig = first_msg.get("content") or ""
                prefix = ""
                if is_session_start:
                    prefix += f"🧠 *Active model: {_MODEL_DISPLAY}*\n\n"
                if ctx_warn_text:
                    prefix += ctx_warn_text
                mem_warning, avail_gb = get_memory_warning()
                if mem_warning:
                    prefix += f"{mem_warning}\n\n"
                if prefix:
                    first_msg["content"] = prefix + orig
                resp_str = json.dumps(resp_data)
                for path, alias in ALIAS_REVERSE.items():
                    resp_str = resp_str.replace(path, alias)
                resp_bytes = resp_str.encode()
                in_tok = usage.get("prompt_tokens", 0)
                out_tok = usage.get("completion_tokens", 0)
            except Exception as e:
                bolog(f"RESPONSE_PARSE_ERROR {e}")
            ctype = resp.headers.get("content-type", "application/json")
            if resp.status_code != 200:
                bolog(f"CHAT_BACKEND_NON200 status={resp.status_code} body={resp_bytes[:300]!r}")
            else:
                bolog(f"CHAT_BACKEND_OK status={resp.status_code}")
            _walter_trace(data.get("model", original_model), body_bytes, in_tok, out_tok, 0,
                          time.monotonic() - start, user_id, user_class, False, error=(resp.status_code != 200))
            return Response(content=resp_bytes, status_code=resp.status_code, media_type=ctype)
    except Exception as e:
        bolog(f"BACKEND_ERROR {e}")
        _walter_trace(data.get("model", original_model), body_bytes, 0, 0, 0,
                      time.monotonic() - start, user_id, user_class, False, error=True)
        return JSONResponse({"error": str(e)}, status_code=502)
    finally:
        c = _dec_inflight(backend)
        bolog(f"INFLIGHT_DEC backend={backend} count={c}")


# Walter catch-all passthrough for non-chat paths (health, models, cache, etc.)
# Registered only in walter mode at startup (see below) so it doesn't shadow the
# alfred routes.
async def walter_passthrough(request: Request):
    if request.method == "GET" and request.url.path.startswith("/v1/models/"):
        model_id = request.url.path[len("/v1/models/"):]
        return JSONResponse({"id": model_id, "object": "model", "created": int(time.time()), "owned_by": "my-local"})
    url = WALTER_BACKEND + _rewrite_backend_path(request.url.path)
    body_bytes = await request.body()
    headers = {k: v for k, v in request.headers.items()
               if k.lower() not in ("host", "content-length", "transfer-encoding")}
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
            resp = await client.request(request.method, url, headers=headers,
                                        content=body_bytes or None, params=dict(request.query_params))
            ctype = resp.headers.get("content-type", "application/json")
            return Response(content=resp.content, status_code=resp.status_code, media_type=ctype)
    except Exception as e:
        bolog(f"BACKEND_ERROR {e}")
        return JSONResponse({"error": str(e)}, status_code=502)


async def cache_janitor():
    while True:
        try:
            await asyncio.sleep(30)
            if sum(_inflight_per_backend.values()) > 0:
                continue
            idle_sec = time.time() - _last_request_time
            if idle_sec < CACHE_IDLE_SECONDS:
                continue
            size_mb = await get_cache_size_mb()
            if size_mb > CACHE_PURGE_MB_THRESHOLD:
                bolog(f"JANITOR_PURGE cache={size_mb:.0f}MB idle={idle_sec:.0f}s")
                await purge_backend_cache()
        except Exception as e:
            bolog(f"JANITOR_ERROR {e}")


# ===========================================================================
# LIFECYCLE
# ===========================================================================
@app.on_event("startup")
async def startup():
    _init_otel()
    _load_state()
    _restore_counters()
    log.info("gordon-gateway started mode=%s port=%s tracing=%s", MODE, PORT, TRACING_ENABLED)

    if TRACING_ENABLED:
        async def _periodic_save():
            while True:
                await asyncio.sleep(STATE_SAVE_INTERVAL)
                _save_state()
        asyncio.create_task(_periodic_save())

    if MODE == "walter":
        # blockops cache janitor + catch-all passthrough route (registered last so
        # the explicit /v1/chat/completions etc. take precedence).
        asyncio.create_task(cache_janitor())
        app.add_api_route("/{path_info:path}", walter_passthrough,
                          methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])


@app.on_event("shutdown")
async def shutdown():
    _save_state()
    if TRACING_ENABLED:
        try:
            _trace_provider.shutdown()
            _metric_provider.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, log_config=LOG_CONFIG)

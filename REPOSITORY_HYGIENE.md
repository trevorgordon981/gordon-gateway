# Source-control boundary

Track `gateway.py`, `config.example.yaml`, `pyproject.toml`, `requirements.txt`,
`scripts/`, documentation, and tests. Never track a deployment-local
`config.yaml`, environment/secret files, PID files, logs, persistent metrics
state, or manual backups. They are deployment-local state and may contain
credentials or operational identifiers.

Secrets (`MASTER_KEY`, `OPENROUTER_API_KEY`) must be injected via explicit
process environment variables or an owner-only (`chmod 0600`) secret file
referenced by `GATEWAY_SECRET_ENV`. Never place credentials in launchd plists,
systemd units, YAML config files, or Git history.

The stdlib-only offline tests in `tests/test_offline.py` and the AST-loaded
tests in `tests/test_max_tokens.py` run with zero third-party packages; the full
suite (including `tests/test_gateway_reliability.py`, `test_gateway_sessions.py`,
and `scripts/demo_offline.py`) exercises end-to-end routing, SSE framing,
failover, and protocol translation completely offline against in-process mocks.

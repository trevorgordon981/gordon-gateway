# Source-control boundary

Track `gateway.py`, `config.example.yaml`, documentation, and tests. Never track
the live `config.yaml`, environment files, PID files, logs, persistent metrics
state, or manual backups. They are deployment-local state and may contain
credentials or operational identifiers.

Studio's gateway directory began as a non-Git deployment. Reconcile this
reviewed source-only tree onto the existing Forgejo repository without
rewriting its history. The deterministic Forgejo job
tests syntax and the exact session-key/LRU helper functions without importing
third-party web libraries, starting a server, contacting a backend, or requiring
a model/GPU. The fuller `test_gateway_sessions.py` remains available for a
dependency-equipped isolated integration run.

"""Exercise production session helpers without importing FastAPI/httpx.

The exact function AST is loaded from gateway.py into a standard-library-only
namespace, so CI needs no packages and cannot start a server or contact a
backend.
"""

from __future__ import annotations

import ast
from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "gateway.py"


class MinimalSessionState:
    def __init__(self, last_seen: float):
        self.last_seen = last_seen


def load_session_helpers():
    tree = ast.parse(SOURCE.read_text(), filename=str(SOURCE))
    names = {"_stable_json", "_session_key", "_prune_session_states", "_get_session_state"}
    assignments = {"_SESSION_ID_HEADERS", "_SESSION_METADATA_KEYS"}
    selected = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
            selected.append(node)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            target_names = {target.id for target in targets if isinstance(target, ast.Name)}
            if target_names & assignments:
                selected.append(node)
    namespace = {
        "Any": Any,
        "OrderedDict": OrderedDict,
        "WalterSessionState": MinimalSessionState,
        "SESSION_STATE_MAX": 2,
        "SESSION_STATE_TTL_SECONDS": 60,
        "_session_states": OrderedDict(),
        "hashlib": hashlib,
        "json": json,
        "time": time,
    }
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace


def load_secret_helper():
    tree = ast.parse(SOURCE.read_text(), filename=str(SOURCE))
    selected = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "_SECRET_ENV_NAMES"
            for target in node.targets
        ):
            selected.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name == "_load_secret_env":
            selected.append(node)
    namespace = {"os": os, "Path": Path}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace["_load_secret_env"]


class SecretFileLoading(unittest.TestCase):
    def test_owner_only_file_loads_allowlist_without_overriding_environment(self):
        load = load_secret_helper()
        with tempfile.TemporaryDirectory() as tmp:
            secret_file = Path(tmp) / "gateway.env"
            secret_file.write_text(
                "MASTER_KEY=from-owner-only-file\n"
                "OPENROUTER_API_KEY='router-from-file'\n"
                "UNRELATED_SECRET=must-not-load\n"
            )
            secret_file.chmod(0o600)
            with patch.dict(
                os.environ,
                {
                    "GATEWAY_SECRET_ENV": str(secret_file),
                    "MASTER_KEY": "explicit-environment-wins",
                },
                clear=True,
            ):
                load()
                self.assertEqual("explicit-environment-wins", os.environ["MASTER_KEY"])
                self.assertEqual("router-from-file", os.environ["OPENROUTER_API_KEY"])
                self.assertNotIn("UNRELATED_SECRET", os.environ)

    def test_group_readable_secret_file_is_rejected(self):
        load = load_secret_helper()
        with tempfile.TemporaryDirectory() as tmp:
            secret_file = Path(tmp) / "gateway.env"
            secret_file.write_text("MASTER_KEY=unsafe-mode\n")
            secret_file.chmod(0o640)
            with patch.dict(
                os.environ, {"GATEWAY_SECRET_ENV": str(secret_file)}, clear=True
            ):
                with self.assertRaises(SystemExit):
                    load()


class SessionIdentity(unittest.TestCase):
    def setUp(self):
        self.ns = load_session_helpers()

    def test_explicit_session_is_stable_scoped_and_opaque(self):
        key = self.ns["_session_key"]
        body = {"messages": [{"role": "user", "content": "hello"}]}
        one, source = key({"X-Claude-Code-Session-Id": "secret-id"}, body, "trevor")
        grown, _ = key(
            {"X-Claude-Code-Session-Id": "secret-id"},
            {"messages": body["messages"] + [{"role": "assistant", "content": "hi"}]},
            "trevor",
        )
        agent, _ = key(
            {"X-Claude-Code-Session-Id": "secret-id", "X-Claude-Code-Agent-Id": "agent-a"},
            body,
            "trevor",
        )
        self.assertEqual(one, grown)
        self.assertNotEqual(one, agent)
        self.assertNotIn("secret-id", one)
        self.assertEqual("x-claude-code-session-id", source)

    def test_fallback_root_survives_conversation_growth(self):
        key = self.ns["_session_key"]
        root = [{"role": "system", "content": "be useful"}, {"role": "user", "content": "root"}]
        one, source = key({}, {"messages": root}, "trevor")
        grown, _ = key({}, {"messages": root + [{"role": "assistant", "content": "answer"}]}, "trevor")
        different, _ = key({}, {"messages": [{"role": "user", "content": "other"}]}, "trevor")
        self.assertEqual(one, grown)
        self.assertNotEqual(one, different)
        self.assertEqual("conversation-root", source)


class BoundedStateStore(unittest.TestCase):
    def setUp(self):
        self.ns = load_session_helpers()

    def test_lru_bound_and_idle_expiry(self):
        get = self.ns["_get_session_state"]
        states = self.ns["_session_states"]
        get("a", now=100)
        get("b", now=101)
        get("a", now=102)
        get("c", now=103)
        self.assertEqual(["a", "c"], list(states))
        get("fresh", now=164)
        self.assertEqual(["fresh"], list(states))


if __name__ == "__main__":
    unittest.main()

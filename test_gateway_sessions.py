"""Focused regression tests for Gordon Gateway's Walter session isolation."""

import json
import os
import unittest
from unittest.mock import AsyncMock, patch


# Import in an entirely local, non-tracing configuration. These values are read
# at module import time; no server is started and no backend is contacted.
os.environ["GATEWAY_MODE"] = "walter"
os.environ["TRACING_ENABLED"] = "false"
os.environ["GATEWAY_CONFIG"] = "/tmp/gordon-gateway-test-no-config.yaml"
os.environ["LOG_FILE"] = "/tmp/gordon-gateway-test.log"
os.environ["BLOCKOPS_LOG_FILE"] = "/tmp/gordon-gateway-blockops-test.log"

from fastapi import Request  # noqa: E402

import gateway  # noqa: E402


def make_request(payload: dict, session_id: str, agent_id: str = "") -> Request:
    body = json.dumps(payload).encode()
    raw_headers = [
        (b"content-type", b"application/json"),
        (b"x-claude-code-session-id", session_id.encode()),
    ]
    if agent_id:
        raw_headers.append((b"x-claude-code-agent-id", agent_id.encode()))

    delivered = False

    async def receive():
        nonlocal delivered
        if delivered:
            return {"type": "http.disconnect"}
        delivered = True
        return {"type": "http.request", "body": body, "more_body": False}

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/chat/completions",
        "raw_path": b"/v1/chat/completions",
        "query_string": b"",
        "root_path": "",
        "headers": raw_headers,
        "client": ("127.0.0.1", 12345),
        "server": ("127.0.0.1", 8080),
    }
    return Request(scope, receive)


class SessionKeyTests(unittest.TestCase):
    def test_claude_session_header_is_stable_and_opaque(self):
        headers = {"X-Claude-Code-Session-Id": "sensitive-session-uuid"}
        short = {"messages": [{"role": "user", "content": "hello"}]}
        grown = {"messages": short["messages"] + [
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "continue"},
        ]}

        key1, source1 = gateway._session_key(headers, short, "trevor")
        key2, source2 = gateway._session_key(headers, grown, "trevor")

        self.assertEqual(key1, key2)
        self.assertEqual(source1, "x-claude-code-session-id")
        self.assertEqual(source2, source1)
        self.assertNotIn("sensitive-session-uuid", key1)
        self.assertNotEqual(
            key1,
            gateway._session_key(
                {"X-Claude-Code-Session-Id": "another-session"}, short, "trevor",
            )[0],
        )

    def test_user_and_agent_both_scope_explicit_session(self):
        body = {"messages": [{"role": "user", "content": "hello"}]}
        base = {"x-claude-code-session-id": "shared-parent"}
        parent = gateway._session_key(base, body, "trevor")[0]
        agent = gateway._session_key(
            {**base, "x-claude-code-agent-id": "agent-a"}, body, "trevor",
        )[0]
        other_agent = gateway._session_key(
            {**base, "x-claude-code-agent-id": "agent-b"}, body, "trevor",
        )[0]
        other_user = gateway._session_key(base, body, "someone-else")[0]

        self.assertEqual(4, len({parent, agent, other_agent, other_user}))

    def test_metadata_id_and_conversation_root_fallback_are_stable(self):
        body = {
            "metadata": {"session_id": "metadata-session"},
            "messages": [{"role": "user", "content": "hello"}],
        }
        metadata_key, source = gateway._session_key({}, body, "trevor")
        self.assertEqual("metadata.session_id", source)
        self.assertTrue(metadata_key.startswith("explicit:"))

        root = [
            {"role": "system", "content": "be useful"},
            {"role": "user", "content": "root prompt"},
        ]
        key1, source1 = gateway._session_key({}, {"messages": root}, "trevor")
        key2, _ = gateway._session_key({}, {"messages": root + [
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": "follow-up"},
        ]}, "trevor")
        key3, _ = gateway._session_key({}, {"messages": [
            {"role": "system", "content": "be useful"},
            {"role": "user", "content": "different root"},
        ]}, "trevor")

        self.assertEqual(key1, key2)
        self.assertNotEqual(key1, key3)
        self.assertEqual("conversation-root", source1)


class SessionStoreTests(unittest.TestCase):
    def setUp(self):
        gateway._session_states.clear()

    def tearDown(self):
        gateway._session_states.clear()

    def test_store_is_lru_bounded(self):
        with patch.object(gateway, "SESSION_STATE_MAX", 2), patch.object(
            gateway, "SESSION_STATE_TTL_SECONDS", 1_000,
        ):
            gateway._get_session_state("a", now=100)
            gateway._get_session_state("b", now=101)
            gateway._get_session_state("a", now=102)
            gateway._get_session_state("c", now=103)

        self.assertEqual(["a", "c"], list(gateway._session_states))

    def test_store_expires_idle_sessions(self):
        with patch.object(gateway, "SESSION_STATE_TTL_SECONDS", 60):
            gateway._get_session_state("old", now=100)
            gateway._get_session_state("new", now=161)

        self.assertNotIn("old", gateway._session_states)
        self.assertIn("new", gateway._session_states)

    def test_parse_errors_and_cache_accounting_are_per_session_and_bounded(self):
        first = gateway.WalterSessionState()
        second = gateway.WalterSessionState()

        gateway.parse_tool_calls("TOOL_CALL: definitely-not-json", first.parse_error_counts)
        gateway.parse_tool_calls("TOOL_CALL: definitely-not-json", first.parse_error_counts)
        self.assertEqual([2], list(first.parse_error_counts.values()))
        self.assertEqual({}, second.parse_error_counts)

        with patch.object(gateway, "SESSION_PARSE_ERROR_KEYS_MAX", 2):
            gateway._record_parse_error(first.parse_error_counts, "second")
            gateway._record_parse_error(first.parse_error_counts, "third")
        self.assertEqual(2, len(first.parse_error_counts))
        self.assertNotIn("definitely-not-json", first.parse_error_counts)

        self.assertEqual((0, 100), gateway.estimate_cache_tokens(100, first))
        self.assertEqual((0, 200), gateway.estimate_cache_tokens(200, second))
        self.assertEqual((150, 0), gateway.estimate_cache_tokens(150, first))


class HandlerIsolationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        gateway._session_states.clear()
        gateway._inflight_per_backend.clear()

    def tearDown(self):
        gateway._session_states.clear()
        gateway._inflight_per_backend.clear()

    async def test_reset_clears_only_target_session_and_never_purges_shared_cache(self):
        body = {"model": "qwen35", "messages": [{"role": "user", "content": "new root"}]}
        key_a, _ = gateway._session_key(
            {"x-claude-code-session-id": "session-a"}, body, gateway.PRIMARY_USER_ID,
        )
        key_b, _ = gateway._session_key(
            {"x-claude-code-session-id": "session-b"}, body, gateway.PRIMARY_USER_ID,
        )
        state_a = gateway._get_session_state(key_a)
        state_a.last_message_count = 8
        state_a.banner_sent = True
        state_a.session_nudge_count = 5
        state_a.parse_error_counts["bad-a"] = 2
        state_a.last_actual_prompt_tokens = 90_000
        state_a.is_new_session = False

        state_b = gateway._get_session_state(key_b)
        state_b.last_message_count = 6
        state_b.banner_sent = True
        state_b.session_nudge_count = 3
        state_b.parse_error_counts["bad-b"] = 1
        state_b.last_actual_prompt_tokens = 70_000
        state_b.is_new_session = False

        request = make_request(body, "session-a")
        downstream = AsyncMock(return_value="isolated-response")
        purge = AsyncMock()
        with patch.object(gateway, "_walter_no_tools_path", downstream), \
             patch.object(gateway, "purge_backend_cache", purge), \
             patch.object(gateway, "get_memory_warning", return_value=(None, 64.0)), \
             patch.object(gateway, "bolog"):
            result = await gateway.walter_handle(request)

        self.assertEqual("isolated-response", result)
        purge.assert_not_awaited()
        self.assertEqual(0, state_a.session_nudge_count)
        self.assertEqual({}, state_a.parse_error_counts)
        self.assertEqual(0, state_a.last_actual_prompt_tokens)
        self.assertEqual(1, state_a.last_message_count)
        self.assertTrue(state_a.banner_sent)
        self.assertTrue(state_a.is_new_session)

        self.assertEqual(6, state_b.last_message_count)
        self.assertEqual(3, state_b.session_nudge_count)
        self.assertEqual({"bad-b": 1}, state_b.parse_error_counts)
        self.assertEqual(70_000, state_b.last_actual_prompt_tokens)
        self.assertFalse(state_b.is_new_session)

    async def test_actual_loop_nudge_is_charged_only_to_its_session(self):
        repeated_call = {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call-any",
                "type": "function",
                "function": {"name": "lookup", "arguments": '{"symbol":"AAPL"}'},
            }],
        }
        looping_body = {
            "model": "qwen35",
            "messages": [
                {"role": "user", "content": "look it up"},
                *[dict(repeated_call) for _ in range(gateway.LOOP_DETECT_NUDGE_THRESHOLD)],
            ],
        }
        quiet_body = {
            "model": "qwen35",
            "messages": [{"role": "user", "content": "unrelated"}],
        }
        downstream = AsyncMock(return_value="isolated-response")
        with patch.object(gateway, "_walter_no_tools_path", downstream), \
             patch.object(gateway, "get_memory_warning", return_value=(None, 64.0)), \
             patch.object(gateway, "bolog"):
            await gateway.walter_handle(make_request(looping_body, "looping-session"))
            await gateway.walter_handle(make_request(quiet_body, "quiet-session"))

        looping_key, _ = gateway._session_key(
            {"x-claude-code-session-id": "looping-session"},
            looping_body,
            gateway.PRIMARY_USER_ID,
        )
        quiet_key, _ = gateway._session_key(
            {"x-claude-code-session-id": "quiet-session"},
            quiet_body,
            gateway.PRIMARY_USER_ID,
        )
        self.assertEqual(1, gateway._session_states[looping_key].session_nudge_count)
        self.assertEqual(0, gateway._session_states[quiet_key].session_nudge_count)


if __name__ == "__main__":
    unittest.main()

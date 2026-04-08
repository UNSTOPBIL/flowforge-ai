"""
E2E Layer 5 – Tool Execution Through the Environment

Tests each tool via env.step() (not directly), verifying both
success and error paths work end-to-end.
"""

import pytest
from flowforge.models import FlowForgeAction


# ── search_db ─────────────────────────────────────────────────────────

class TestSearchDB:
    def test_finds_employees(self, env_easy):
        a = FlowForgeAction(tool_name="search_db", parameters={"query": "engineering"})
        obs, reward, _, _ = env_easy.step(a)
        assert reward > 0
        assert "successfully" in obs.message.lower()

    def test_no_results(self, env_easy):
        a = FlowForgeAction(tool_name="search_db", parameters={"query": "zzzznonexistent"})
        obs, reward, _, _ = env_easy.step(a)
        # No data returned → no progress made → base reward only
        assert isinstance(reward, float)

    def test_missing_query_fails(self, env_easy):
        a = FlowForgeAction(tool_name="search_db", parameters={})
        obs, reward, _, _ = env_easy.step(a)
        assert reward < 0
        assert "error" in obs.message.lower()


# ── send_email ────────────────────────────────────────────────────────

class TestSendEmail:
    def test_valid_email(self, env_medium):
        a = FlowForgeAction(tool_name="send_email", parameters={
            "to": "test@example.com", "subject": "Hi", "body": "Hello"
        })
        obs, reward, _, _ = env_medium.step(a)
        assert reward > 0

    def test_invalid_email_format(self, env_medium):
        a = FlowForgeAction(tool_name="send_email", parameters={
            "to": "not-an-email", "subject": "Hi", "body": "Hello"
        })
        obs, reward, _, _ = env_medium.step(a)
        assert reward < 0

    def test_missing_fields(self, env_medium):
        a = FlowForgeAction(tool_name="send_email", parameters={"to": "a@b.com"})
        obs, reward, _, _ = env_medium.step(a)
        assert reward < 0


# ── read_file ─────────────────────────────────────────────────────────

class TestReadFile:
    def test_existing_file(self, env_hard):
        a = FlowForgeAction(tool_name="read_file", parameters={"file_path": "/reports/sales_summary.txt"})
        obs, reward, _, _ = env_hard.step(a)
        assert reward > 0

    def test_nonexistent_file(self, env_hard):
        a = FlowForgeAction(tool_name="read_file", parameters={"file_path": "/ghost.txt"})
        obs, reward, _, _ = env_hard.step(a)
        assert reward < 0

    def test_missing_path(self, env_hard):
        a = FlowForgeAction(tool_name="read_file", parameters={})
        obs, reward, _, _ = env_hard.step(a)
        assert reward < 0


# ── run_query ─────────────────────────────────────────────────────────

class TestRunQuery:
    def test_select_all(self, env_hard):
        a = FlowForgeAction(tool_name="run_query", parameters={"query": "SELECT * FROM employees"})
        obs, reward, _, _ = env_hard.step(a)
        assert reward > 0

    def test_with_where(self, env_hard):
        a = FlowForgeAction(tool_name="run_query", parameters={
            "query": "SELECT * FROM employees WHERE department = 'Engineering'"
        })
        obs, reward, _, _ = env_hard.step(a)
        assert reward > 0

    def test_invalid_table(self, env_hard):
        a = FlowForgeAction(tool_name="run_query", parameters={"query": "SELECT * FROM ghosts"})
        obs, reward, _, _ = env_hard.step(a)
        assert reward < 0

    def test_non_select_rejected(self, env_hard):
        a = FlowForgeAction(tool_name="run_query", parameters={"query": "DROP TABLE employees"})
        obs, reward, _, _ = env_hard.step(a)
        assert reward < 0

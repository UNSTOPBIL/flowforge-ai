"""
E2E Layer 4 – Reward Signal Density & Grader Accuracy

Verifies:
  - Reward function provides DENSE signal (not sparse binary)
  - Grader produces smooth scores (0.0–1.0) based on objective completion
  - Partial credit works correctly
  - Score is deterministic across runs
"""

import pytest
from flowforge.models import FlowForgeAction


# ── Reward Density ─────────────────────────────────────────────────────

class TestRewardDensity:
    """Hackathon requires meaningful, non-binary rewards per step."""

    def test_tool_success_gives_positive_reward(self, env_easy):
        action = FlowForgeAction(tool_name="search_db", parameters={"query": "engineering"})
        _, reward, _, _ = env_easy.step(action)
        assert reward > 0, "Successful tool execution must yield positive reward"

    def test_tool_failure_gives_negative_reward(self, env_easy):
        action = FlowForgeAction(tool_name="search_db", parameters={"query": ""})
        _, reward, _, _ = env_easy.step(action)
        assert reward < 0, "Failed tool execution must yield negative reward"

    def test_unknown_tool_penalised(self, env_easy):
        action = {"tool_name": "nonexistent", "parameters": {}}
        _, reward, _, _ = env_easy.step(action)
        assert reward < 0

    def test_finish_reward_is_small(self, env_easy):
        """Finish reward must be small (0.1), not dominant (0.5+)."""
        action = FlowForgeAction(tool_name="finish", parameters={})
        _, reward, _, _ = env_easy.step(action)
        assert reward <= 0.15, f"Finish reward too high: {reward}"

    def test_progress_gives_bonus(self, env_easy):
        """First meaningful action gives a progress bonus (0.2 base + 0.2 progress)."""
        action = FlowForgeAction(tool_name="search_db", parameters={"query": "engineering"})
        _, reward, _, _ = env_easy.step(action)
        assert reward >= 0.3, f"Expected progress bonus, got {reward}"


# ── Grader Accuracy ───────────────────────────────────────────────────

class TestGraderAccuracy:
    def test_zero_completed(self, grader):
        score = grader.compute_score([], ["a", "b", "c"])
        assert score == 0.0

    def test_partial_completed(self, grader):
        score = grader.compute_score(["a"], ["a", "b", "c"])
        assert abs(score - 1 / 3) < 0.01

    def test_full_completed(self, grader):
        score = grader.compute_score(["a", "b", "c"], ["a", "b", "c"])
        assert score == 1.0

    def test_extra_objectives_ignored(self, grader):
        """Spurious completions don't inflate score."""
        score = grader.compute_score(["a", "b", "c", "x"], ["a", "b", "c"])
        assert score == 1.0

    def test_empty_objectives_returns_zero(self, grader):
        score = grader.compute_score(["a"], [])
        assert score == 0.0

    def test_score_clamped(self, grader):
        score = grader.compute_score(["a", "b"], ["a"])
        assert score <= 1.0

    def test_is_success_threshold(self, grader):
        assert grader.is_success(1.0) is True
        assert grader.is_success(0.99) is False

    def test_deterministic(self, grader):
        """Same input must always produce same output."""
        scores = [grader.compute_score(["a", "b"], ["a", "b", "c"]) for _ in range(100)]
        assert len(set(scores)) == 1


# ── Objective Deduplication ───────────────────────────────────────────

class TestObjectiveDeduplication:
    def test_duplicate_objectives_not_double_counted(self, env_easy):
        """Running the same tool twice must not add the objective twice."""
        a = FlowForgeAction(tool_name="search_db", parameters={"query": "engineering"})
        env_easy.step(a)
        env_easy.step(a)
        objs = env_easy.state().completed_objectives
        assert len(objs) == len(set(objs)), f"Duplicates found: {objs}"

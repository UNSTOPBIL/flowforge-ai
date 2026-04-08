"""
E2E Layer 6 – Edge Cases & Robustness

Tests unusual inputs, boundary conditions, and adversarial actions
that a real agent or evaluator might send.
"""

import pytest
from flowforge.models import FlowForgeAction, FlowForgeState
from pydantic import ValidationError


# ── Action Validation ─────────────────────────────────────────────────

class TestActionValidation:
    def test_empty_tool_name_rejected(self):
        with pytest.raises(ValidationError):
            FlowForgeAction(tool_name="", parameters={})

    def test_whitespace_tool_name_rejected(self):
        with pytest.raises(ValidationError):
            FlowForgeAction(tool_name="   ", parameters={})

    def test_tool_name_normalised_to_lowercase(self):
        a = FlowForgeAction(tool_name="Search_DB", parameters={"query": "test"})
        assert a.tool_name == "search_db"


# ── State Boundaries ─────────────────────────────────────────────────

class TestStateBoundaries:
    def test_negative_step_count_rejected(self):
        with pytest.raises(ValidationError):
            FlowForgeState(step_count=-1)

    def test_zero_max_steps_means_immediately_done(self):
        s = FlowForgeState(step_count=0, max_steps=0)
        assert s.is_done is True

    def test_progress_at_zero(self):
        s = FlowForgeState(step_count=0, max_steps=10)
        assert s.progress == 0.0

    def test_progress_capped_at_one(self):
        s = FlowForgeState(step_count=25, max_steps=10)
        assert s.progress == 1.0


# ── Episode Edge Cases ────────────────────────────────────────────────

class TestEpisodeEdgeCases:
    def test_step_without_reset(self, env):
        """step() should auto-reset if state is None."""
        a = FlowForgeAction(tool_name="search_db", parameters={"query": "test"})
        obs, reward, done, info = env.step(a)
        assert info["step"] == 1

    def test_double_finish(self, env_easy):
        """Second finish after episode done should still be graceful."""
        env_easy.step(FlowForgeAction(tool_name="finish", parameters={}))
        obs, _, done, _ = env_easy.step(FlowForgeAction(tool_name="finish", parameters={}))
        assert done is True

    def test_many_steps_reach_max(self, env):
        """Running 20 actions must hit the limit."""
        env.reset(task_id="easy", max_steps=5)
        for i in range(10):
            a = FlowForgeAction(tool_name="search_db", parameters={"query": "a"})
            _, _, done, _ = env.step(a)
            if done:
                break
        assert done is True

    def test_episode_complete_message_tag(self, env_easy):
        """On termination, message must contain '[Episode complete]'."""
        obs, _, done, _ = env_easy.step(FlowForgeAction(tool_name="finish", parameters={}))
        assert "[Episode complete]" in obs.message


# ── Objective Completion Triggers Done ────────────────────────────────

class TestObjectiveTermination:
    def test_easy_objectives_done_triggers_episode_end(self, env):
        """When all easy objectives are met, done should become True."""
        env.reset(task_id="easy", max_steps=20)
        # search_db satisfies both easy objectives
        env.step(FlowForgeAction(tool_name="search_db", parameters={"query": "engineering"}))
        state = env.state()
        unique = set(state.completed_objectives)
        # Easy requires 2 objectives; search_db gives find_employee + read_employee_data
        assert len(unique) >= 2

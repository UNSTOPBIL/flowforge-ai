"""
E2E Layer 3 – Environment Lifecycle & API Signature Compliance

Tests that reset(), step(), state(), and close() conform to the OpenEnv
spec signatures and return the correct types.
"""

import pytest
from flowforge.models import FlowForgeAction, FlowForgeObservation, FlowForgeState


# ── reset() ───────────────────────────────────────────────────────────

class TestReset:
    def test_returns_observation(self, env):
        obs = env.reset(task_id="easy")
        assert isinstance(obs, FlowForgeObservation)

    def test_clears_previous_state(self, env):
        env.reset(task_id="easy")
        env.step(FlowForgeAction(tool_name="search_db", parameters={"query": "alice"}))
        obs = env.reset(task_id="easy")
        state = env.state()
        assert state.step_count == 0
        assert state.completed_objectives == []

    def test_accepts_seed_and_episode_id(self, env):
        """Validator passes seed= and episode_id= — must not crash."""
        obs = env.reset(seed=42, episode_id="test-ep-001", task_id="easy")
        assert isinstance(obs, FlowForgeObservation)

    def test_accepts_kwargs(self, env):
        obs = env.reset(task_id="easy", max_steps=5, some_future_param=True)
        assert isinstance(obs, FlowForgeObservation)

    def test_observation_lists_all_tools(self, env):
        obs = env.reset(task_id="easy")
        expected = {"search_db", "send_email", "read_file", "run_query", "schedule_meeting"}
        assert set(obs.available_tools) == expected


# ── step() ────────────────────────────────────────────────────────────

class TestStep:
    def test_returns_four_tuple(self, env_easy):
        action = FlowForgeAction(tool_name="search_db", parameters={"query": "alice"})
        result = env_easy.step(action)
        assert len(result) == 4

    def test_returns_correct_types(self, env_easy):
        action = FlowForgeAction(tool_name="search_db", parameters={"query": "alice"})
        obs, reward, done, info = env_easy.step(action)
        assert isinstance(obs, FlowForgeObservation)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_accepts_dict_action(self, env_easy):
        obs, reward, done, info = env_easy.step({"tool_name": "search_db", "parameters": {"query": "bob"}})
        assert isinstance(obs, FlowForgeObservation)

    def test_accepts_timeout_s(self, env_easy):
        """Validator passes timeout_s= — must not crash."""
        action = FlowForgeAction(tool_name="search_db", parameters={"query": "alice"})
        obs, reward, done, info = env_easy.step(action, timeout_s=10.0)
        assert isinstance(obs, FlowForgeObservation)

    def test_accepts_kwargs(self, env_easy):
        action = FlowForgeAction(tool_name="search_db", parameters={"query": "alice"})
        obs, reward, done, info = env_easy.step(action, some_future_param=True)
        assert isinstance(obs, FlowForgeObservation)

    def test_increments_step_count(self, env_easy):
        action = FlowForgeAction(tool_name="search_db", parameters={"query": "alice"})
        env_easy.step(action)
        assert env_easy.state().step_count == 1
        env_easy.step(action)
        assert env_easy.state().step_count == 2

    def test_finish_sets_done_true(self, env_easy):
        action = FlowForgeAction(tool_name="finish", parameters={})
        _, _, done, _ = env_easy.step(action)
        assert done is True

    def test_unknown_tool_negative_reward(self, env_easy):
        action = {"tool_name": "hack_mainframe", "parameters": {}}
        _, reward, _, _ = env_easy.step(action)
        assert reward < 0

    def test_valid_tool_positive_reward(self, env_easy):
        action = FlowForgeAction(tool_name="search_db", parameters={"query": "engineering"})
        _, reward, _, _ = env_easy.step(action)
        assert reward > 0

    def test_max_steps_terminates(self, env):
        env.reset(task_id="easy", max_steps=2)
        a = FlowForgeAction(tool_name="search_db", parameters={"query": "a"})
        env.step(a)
        _, _, done, _ = env.step(a)
        assert done is True


# ── state() ───────────────────────────────────────────────────────────

class TestState:
    def test_returns_state_object(self, env_easy):
        assert isinstance(env_easy.state(), FlowForgeState)

    def test_auto_resets_if_none(self, env):
        state = env.state()
        assert isinstance(state, FlowForgeState)

    def test_task_id_matches(self, env):
        env.reset(task_id="hard")
        assert env.state().task_id == "hard"


# ── close() ───────────────────────────────────────────────────────────

class TestClose:
    def test_close_clears_state(self, env_easy):
        env_easy.close()
        assert env_easy._state is None

    def test_state_auto_recovers_after_close(self, env_easy):
        env_easy.close()
        state = env_easy.state()
        assert isinstance(state, FlowForgeState)

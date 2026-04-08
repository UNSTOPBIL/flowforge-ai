"""
E2E Layer 1 – Full Episode Runs

Runs the exact same action sequences that inference.py uses for each
difficulty level and asserts that the grader returns a perfect 1.0 score.
This is the single most important test: if it fails, the submission
cannot produce a reproducible baseline.

NOTE: The environment dynamically terminates when all objectives are
completed (via _check_all_objectives_done), so episodes may finish
*before* the explicit `finish` action is reached. This is correct
behaviour — not a bug.
"""

import pytest
from flowforge.models import FlowForgeAction


# ── Deterministic action sequences (mirrors inference.py) ─────────────

EASY_ACTIONS = [
    {"tool_name": "search_db", "parameters": {"query": "engineering"}},
    {"tool_name": "finish", "parameters": {}},
]

MEDIUM_ACTIONS = [
    {"tool_name": "search_db", "parameters": {"query": "engineering"}},
    {"tool_name": "send_email", "parameters": {"to": "test@test.com", "subject": "Done", "body": "Task complete"}},
    {"tool_name": "finish", "parameters": {}},
]

HARD_ACTIONS = [
    {"tool_name": "read_file", "parameters": {"file_path": "/reports/complaints_summary.txt"}},
    {"tool_name": "run_query", "parameters": {"query": "SELECT * FROM employees"}},
    {"tool_name": "schedule_meeting", "parameters": {"attendees": ["manager@test.com"], "date": "2024-05-01", "title": "Report Review"}},
    {"tool_name": "send_email", "parameters": {"to": "manager@test.com", "subject": "Report", "body": "See attached"}},
    {"tool_name": "finish", "parameters": {}},
]


def _run_episode(env, task, grader, actions):
    """Helper: play a full episode and return (score, rewards, num_steps, done_early).

    done_early is True if the env terminated before all scripted actions ran
    (i.e., dynamic objective completion triggered).
    """
    env.reset(task_id=task.task_id, max_steps=20)
    objectives = task.get_objectives()
    rewards = []
    done_early = False

    for i, action_dict in enumerate(actions):
        action = FlowForgeAction.model_validate(action_dict)
        obs, reward, done, info = env.step(action)
        rewards.append(reward)
        if done:
            if i < len(actions) - 1:
                done_early = True
            break

    completed = task.check_completion(env.state())
    score = grader.compute_score(completed, objectives)
    return score, rewards, len(rewards), done_early


# ── Tests ─────────────────────────────────────────────────────────────

class TestEasyEpisode:
    def test_perfect_score(self, env, easy_task, grader):
        score, _, _, _ = _run_episode(env, easy_task, grader, EASY_ACTIONS)
        assert score == 1.0, f"Easy task must score 1.0 — got {score}"

    def test_all_rewards_positive(self, env, easy_task, grader):
        _, rewards, _, _ = _run_episode(env, easy_task, grader, EASY_ACTIONS)
        assert all(r >= 0 for r in rewards), f"Rewards should be non-negative: {rewards}"

    def test_episode_terminates_within_budget(self, env, easy_task, grader):
        _, _, steps, _ = _run_episode(env, easy_task, grader, EASY_ACTIONS)
        assert steps <= len(EASY_ACTIONS), f"Easy used {steps} steps, budget was {len(EASY_ACTIONS)}"


class TestMediumEpisode:
    def test_perfect_score(self, env, medium_task, grader):
        score, _, _, _ = _run_episode(env, medium_task, grader, MEDIUM_ACTIONS)
        assert score == 1.0, f"Medium task must score 1.0 — got {score}"

    def test_all_rewards_positive(self, env, medium_task, grader):
        _, rewards, _, _ = _run_episode(env, medium_task, grader, MEDIUM_ACTIONS)
        assert all(r >= 0 for r in rewards), f"Rewards should be non-negative: {rewards}"

    def test_episode_terminates_within_budget(self, env, medium_task, grader):
        _, _, steps, _ = _run_episode(env, medium_task, grader, MEDIUM_ACTIONS)
        assert steps <= len(MEDIUM_ACTIONS)


class TestHardEpisode:
    def test_perfect_score(self, env, hard_task, grader):
        score, _, _, _ = _run_episode(env, hard_task, grader, HARD_ACTIONS)
        assert score == 1.0, f"Hard task must score 1.0 — got {score}"

    def test_all_rewards_positive(self, env, hard_task, grader):
        _, rewards, _, _ = _run_episode(env, hard_task, grader, HARD_ACTIONS)
        assert all(r >= 0 for r in rewards), f"Rewards should be non-negative: {rewards}"

    def test_episode_terminates_within_budget(self, env, hard_task, grader):
        _, _, steps, _ = _run_episode(env, hard_task, grader, HARD_ACTIONS)
        assert steps <= len(HARD_ACTIONS)

"""
E2E Layer 7 – Determinism & Reproducibility

The hackathon requires reproducible baseline scores. These tests
run the same episode multiple times and assert identical results.
"""

import pytest
from flowforge.models import FlowForgeAction


EASY_ACTIONS = [
    {"tool_name": "search_db", "parameters": {"query": "engineering"}},
    {"tool_name": "finish", "parameters": {}},
]


def _run_easy_episode(env, task, grader):
    env.reset(task_id="easy", max_steps=20)
    rewards = []
    for a in EASY_ACTIONS:
        _, r, done, _ = env.step(FlowForgeAction.model_validate(a))
        rewards.append(r)
        if done:
            break
    completed = task.check_completion(env.state())
    score = grader.compute_score(completed, task.get_objectives())
    return score, tuple(rewards)


class TestDeterminism:
    def test_scores_identical_across_10_runs(self, env, easy_task, grader):
        results = [_run_easy_episode(env, easy_task, grader) for _ in range(10)]
        scores = [r[0] for r in results]
        assert len(set(scores)) == 1, f"Non-deterministic scores: {scores}"

    def test_rewards_identical_across_10_runs(self, env, easy_task, grader):
        results = [_run_easy_episode(env, easy_task, grader) for _ in range(10)]
        reward_seqs = [r[1] for r in results]
        assert len(set(reward_seqs)) == 1, f"Non-deterministic rewards: {reward_seqs}"

    def test_reset_truly_clears_state(self, env, easy_task, grader):
        """After a full episode + reset, a fresh run must produce the same result."""
        s1, r1 = _run_easy_episode(env, easy_task, grader)
        s2, r2 = _run_easy_episode(env, easy_task, grader)
        assert s1 == s2 and r1 == r2

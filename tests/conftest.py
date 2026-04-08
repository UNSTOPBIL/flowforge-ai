"""
FlowForge AI – E2E Test Fixtures

Shared fixtures used across every E2E test module.
"""

import pytest
from flowforge.env import FlowForgeEnvironment
from flowforge.grader import FlowForgeGrader
from flowforge.models import FlowForgeAction, FlowForgeObservation, FlowForgeState
from flowforge.tasks.task_easy import EasyTask
from flowforge.tasks.task_medium import MediumTask
from flowforge.tasks.task_hard import HardTask


# ── Environment & Grader ──────────────────────────────────────────────

@pytest.fixture
def env():
    """Fresh FlowForgeEnvironment instance."""
    return FlowForgeEnvironment()


@pytest.fixture
def grader():
    """FlowForgeGrader instance."""
    return FlowForgeGrader()


# ── Task Instances ────────────────────────────────────────────────────

@pytest.fixture
def easy_task():
    return EasyTask()


@pytest.fixture
def medium_task():
    return MediumTask()


@pytest.fixture
def hard_task():
    return HardTask()


# ── Pre-Reset Environments ───────────────────────────────────────────

@pytest.fixture
def env_easy(env):
    """Environment reset for the easy task."""
    env.reset(task_id="easy", max_steps=20)
    return env


@pytest.fixture
def env_medium(env):
    """Environment reset for the medium task."""
    env.reset(task_id="medium", max_steps=20)
    return env


@pytest.fixture
def env_hard(env):
    """Environment reset for the hard task."""
    env.reset(task_id="hard", max_steps=20)
    return env

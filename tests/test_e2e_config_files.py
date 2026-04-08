"""
E2E Layer 8 – Configuration & Deployment File Validation

Validates that openenv.yaml, Dockerfile, pyproject.toml, and README.md
meet the hackathon submission requirements without needing Docker.
"""

import os
import yaml
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestOpenenvYaml:
    def _load(self):
        path = os.path.join(PROJECT_ROOT, "openenv.yaml")
        assert os.path.isfile(path), "openenv.yaml not found in project root"
        with open(path) as f:
            return yaml.safe_load(f)

    def test_has_name(self):
        assert self._load()["name"] == "flowforge-ai"

    def test_has_version(self):
        assert "version" in self._load()

    def test_has_entry_point(self):
        ep = self._load().get("entry_point", "")
        assert "FlowForgeEnvironment" in ep

    def test_has_action_model(self):
        assert "FlowForgeAction" in self._load().get("action_model", "")

    def test_has_observation_model(self):
        assert "FlowForgeObservation" in self._load().get("observation_model", "")

    def test_has_three_tasks(self):
        tasks = self._load().get("tasks", [])
        assert len(tasks) == 3
        ids = {t["id"] for t in tasks}
        assert ids == {"easy", "medium", "hard"}


class TestDockerfile:
    def _read(self):
        path = os.path.join(PROJECT_ROOT, "Dockerfile")
        assert os.path.isfile(path), "Dockerfile not found in project root"
        with open(path) as f:
            return f.read()

    def test_exposes_7860(self):
        assert "7860" in self._read()

    def test_has_healthcheck(self):
        assert "HEALTHCHECK" in self._read()

    def test_installs_curl(self):
        content = self._read()
        assert "curl" in content, "Dockerfile must install curl for HEALTHCHECK"

    def test_sets_pythonpath(self):
        assert "PYTHONPATH" in self._read()


class TestInferenceFileLocation:
    def test_inference_py_in_root(self):
        path = os.path.join(PROJECT_ROOT, "inference.py")
        assert os.path.isfile(path), "inference.py must be in the project root"


class TestReadme:
    def _read(self):
        path = os.path.join(PROJECT_ROOT, "README.md")
        assert os.path.isfile(path), "README.md not found"
        with open(path) as f:
            return f.read()

    def test_has_hf_frontmatter(self):
        content = self._read()
        assert content.startswith("---"), "README must start with YAML front-matter for HF Spaces"
        assert "app_port: 7860" in content

    def test_mentions_action_space(self):
        assert "Action Space" in self._read()

    def test_mentions_observation_space(self):
        assert "Observation Space" in self._read()

    def test_mentions_reward_function(self):
        assert "Reward" in self._read()

    def test_mentions_baseline_scores(self):
        assert "Baseline" in self._read()

    def test_has_docker_instructions(self):
        assert "docker" in self._read().lower()

    def test_minimum_length(self):
        """README must be substantial (>500 chars), not a stub."""
        assert len(self._read()) > 500

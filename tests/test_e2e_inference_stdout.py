"""
E2E Layer 2 – Inference Script Structured Logging Compliance

Runs inference.py as a subprocess (exactly like the hackathon evaluator does)
and validates every line of stdout against the mandatory parsing format.

Spec:
  [START] task=<id> env=flowforge-ai model=<name>
  [STEP]  step=<n> action=<json> reward=<float> done=<bool> error=<msg|null>
  [END]   success=<bool> steps=<n> score=<float> rewards=<csv>
"""

import os
import re
import sys
import subprocess


def _run_inference():
    """Execute inference.py as subprocess and capture stdout."""
    env = os.environ.copy()
    env["PYTHONPATH"] = env.get("PYTHONPATH", "") + ":."
    result = subprocess.run(
        [sys.executable, "inference.py"],
        capture_output=True,
        text=True,
        cwd=".",
        env=env,
    )
    assert result.returncode == 0, f"inference.py crashed:\n{result.stderr}"
    return result.stdout


# ── [START] Tag ───────────────────────────────────────────────────────

class TestStartTag:
    def test_exactly_three_start_tags(self):
        output = _run_inference()
        starts = [l for l in output.splitlines() if l.startswith("[START]")]
        assert len(starts) == 3, f"Expected 3 [START] lines, got {len(starts)}"

    def test_format_matches_spec(self):
        output = _run_inference()
        pattern = r"^\[START\] task=\w+ env=flowforge-ai model=[\w\-]+$"
        for line in output.splitlines():
            if line.startswith("[START]"):
                assert re.match(pattern, line), f"Bad [START] format: {line}"

    def test_task_ids_present(self):
        output = _run_inference()
        starts = [l for l in output.splitlines() if l.startswith("[START]")]
        ids = {re.search(r"task=(\w+)", l).group(1) for l in starts}
        assert ids == {"easy", "medium", "hard"}, f"Missing task IDs: {ids}"


# ── [STEP] Tag ────────────────────────────────────────────────────────

class TestStepTag:
    def test_step_tags_exist(self):
        output = _run_inference()
        steps = [l for l in output.splitlines() if l.startswith("[STEP]")]
        assert len(steps) >= 3, "Must have at least 3 [STEP] lines (1 per task min)"

    def test_format_matches_spec(self):
        output = _run_inference()
        pattern = r"^\[STEP\] step=\d+ action=.+ reward=-?\d+\.\d{2} done=(true|false) error=.+$"
        for line in output.splitlines():
            if line.startswith("[STEP]"):
                assert re.match(pattern, line), f"Bad [STEP] format: {line}"

    def test_reward_is_two_decimal_places(self):
        output = _run_inference()
        for line in output.splitlines():
            if line.startswith("[STEP]"):
                m = re.search(r"reward=(-?\d+\.\d+)", line)
                assert m, f"No reward field found: {line}"
                assert len(m.group(1).split(".")[-1]) == 2, f"Reward not .2f: {line}"

    def test_done_is_lowercase_boolean(self):
        output = _run_inference()
        for line in output.splitlines():
            if line.startswith("[STEP]"):
                m = re.search(r"done=(true|false)", line)
                assert m, f"done must be lowercase 'true'/'false': {line}"

    def test_no_python_booleans(self):
        output = _run_inference()
        assert " True " not in output, "Python-cased 'True' found in stdout"
        assert " False " not in output, "Python-cased 'False' found in stdout"


# ── [END] Tag ─────────────────────────────────────────────────────────

class TestEndTag:
    def test_exactly_three_end_tags(self):
        output = _run_inference()
        ends = [l for l in output.splitlines() if l.startswith("[END]")]
        assert len(ends) == 3, f"Expected 3 [END] lines, got {len(ends)}"

    def test_format_matches_spec(self):
        output = _run_inference()
        pattern = r"^\[END\] success=(true|false) steps=\d+ score=\d+\.\d{2} rewards=.+$"
        for line in output.splitlines():
            if line.startswith("[END]"):
                assert re.match(pattern, line), f"Bad [END] format: {line}"

    def test_rewards_no_square_brackets(self):
        """Parser will reject rewards=[...] — must be comma-separated."""
        output = _run_inference()
        for line in output.splitlines():
            if line.startswith("[END]"):
                m = re.search(r"rewards=(.*)", line)
                assert m, f"No rewards field: {line}"
                rewards_str = m.group(1)
                assert "[" not in rewards_str, f"Square brackets in rewards: {line}"
                assert "]" not in rewards_str, f"Square brackets in rewards: {line}"

    def test_rewards_are_csv_floats(self):
        output = _run_inference()
        for line in output.splitlines():
            if line.startswith("[END]"):
                m = re.search(r"rewards=(.*)", line)
                parts = m.group(1).split(",")
                for p in parts:
                    float(p.strip())  # must not raise ValueError

    def test_score_in_0_to_1(self):
        output = _run_inference()
        for line in output.splitlines():
            if line.startswith("[END]"):
                m = re.search(r"score=(\d+\.\d+)", line)
                score = float(m.group(1))
                assert 0.0 <= score <= 1.0, f"Score out of range: {score}"


# ── Cross-Tag Consistency ─────────────────────────────────────────────

class TestCrossTagConsistency:
    def test_start_end_pairing(self):
        """Every [START] must have a matching [END]."""
        output = _run_inference()
        starts = len([l for l in output.splitlines() if l.startswith("[START]")])
        ends = len([l for l in output.splitlines() if l.startswith("[END]")])
        assert starts == ends, f"Mismatched START/END: {starts} vs {ends}"

    def test_end_steps_matches_step_count(self):
        """steps= in [END] must equal the number of [STEP] lines for that episode."""
        output = _run_inference()
        episodes = []
        current_steps = 0
        for line in output.splitlines():
            if line.startswith("[START]"):
                current_steps = 0
            elif line.startswith("[STEP]"):
                current_steps += 1
            elif line.startswith("[END]"):
                m = re.search(r"steps=(\d+)", line)
                reported = int(m.group(1))
                assert reported == current_steps, (
                    f"[END] says steps={reported} but counted {current_steps} [STEP] lines"
                )

    def test_end_rewards_count_matches_steps(self):
        """Number of comma-separated rewards must equal steps."""
        output = _run_inference()
        for line in output.splitlines():
            if line.startswith("[END]"):
                m_steps = re.search(r"steps=(\d+)", line)
                m_rewards = re.search(r"rewards=(.*)", line)
                steps = int(m_steps.group(1))
                rewards = m_rewards.group(1).split(",")
                assert len(rewards) == steps, (
                    f"rewards count ({len(rewards)}) != steps ({steps})"
                )

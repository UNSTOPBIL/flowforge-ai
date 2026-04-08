"""
FlowForge AI - Grader

Deterministic scoring system that evaluates task completion
based on the ratio of completed objectives to total objectives.
"""

from __future__ import annotations


class FlowForgeGrader:
    """Computes task scores and determines success status.

    Scoring is fully deterministic with no randomness.
    Returns a smooth score between 0.0 and 1.0 based on
    the proportion of completed objectives.
    """

    def compute_score(self, completed_objectives: list[str], total_objectives: list[str]) -> float:
        """Compute the completion score.

        Args:
            completed_objectives: List of objective names that were completed.
            total_objectives: List of all objective names for the task.

        Returns:
            Score between 0.0 and 1.0. Returns 0.0 if there are no objectives.
        """
        if not total_objectives:
            return 0.0

        completed_set = set(completed_objectives)
        total_set = set(total_objectives)
        valid_completed = completed_set.intersection(total_set)

        score = len(valid_completed) / len(total_set)
        return min(1.0, max(0.0, score))

    def compute_score_with_efficiency(self, completed_objectives: list[str], total_objectives: list[str], steps_used: int, max_steps: int) -> float:
        """Compute the completion score with an efficiency bonus.

        Args:
            completed_objectives: List of objective names that were completed.
            total_objectives: List of all objective names for the task.
            steps_used: Number of steps used.
            max_steps: Maximum allowed steps.

        Returns:
            Score between 0.0 and 1.0. Efficiency bonus raises the score but is capped at 1.0.
        """
        base_score = self.compute_score(completed_objectives, total_objectives)
        if base_score < 1.0:
            return base_score
        efficiency_bonus = max(0.0, 1.0 - (steps_used / max(1, max_steps)))
        # Keep base at 1.0 max, return exactly 1.0 in case we want to separate bonus logic
        return min(1.0, base_score + efficiency_bonus)

    def is_success(self, score: float) -> bool:
        """Determine if the score represents full success.

        Args:
            score: The computed score from compute_score().

        Returns:
            True if score equals 1.0 (all objectives completed).
        """
        return score == 1.0

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

    def is_success(self, score: float) -> bool:
        """Determine if the score represents full success.

        Args:
            score: The computed score from compute_score().

        Returns:
            True if score equals 1.0 (all objectives completed).
        """
        return score == 1.0

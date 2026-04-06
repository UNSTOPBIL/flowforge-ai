"""
FlowForge AI - Hard Task

Task requiring the agent to read a file, run a database query, and send an email.
All three objectives must be completed to finish the task.
"""

from __future__ import annotations

from models import FlowForgeState


class HardTask:
    """Hard-level task: read a file, run a query, and send an email."""

    task_id = "hard"
    description = "Read a report file, run a database query for analysis, and send the results via email"
    objectives = ["read_file", "run_query", "send_email"]

    def get_objectives(self) -> list[str]:
        """Return the list of task objectives."""
        return list(self.objectives)

    def check_completion(self, state: FlowForgeState) -> list[str]:
        """Check which objectives are completed based on current state.

        Args:
            state: Current FlowForgeState.

        Returns:
            List of completed objective names.
        """
        completed = []
        if state.intermediate_data.get("file_read"):
            completed.append("read_file")
        if state.intermediate_data.get("query_done"):
            completed.append("run_query")
        if state.intermediate_data.get("email_sent"):
            completed.append("send_email")
        return completed

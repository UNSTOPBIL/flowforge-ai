"""
FlowForge AI - Easy Task

Simple task requiring the agent to find an employee in the database.
"""

from __future__ import annotations

try:
    from flowforge.models import FlowForgeAction, FlowForgeObservation, FlowForgeState
except ImportError:
    from models import FlowForgeState


class EasyTask:
    """Easy-level task: find an employee in the database."""

    task_id = "easy"
    description = "Search the database to find an employee by name or department"
    objectives = ["find_employee", "read_employee_data"]

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
        if state.intermediate_data.get("employee_found"):
            completed.append("find_employee")
        if state.intermediate_data.get("employee_data_read"):
            completed.append("read_employee_data")
        return completed

"""
FlowForge AI - Medium Task

Task requiring the agent to find an employee and send them an email.
Objectives must be completed in order.
"""

from __future__ import annotations

try:
    from flowforge.models import FlowForgeAction, FlowForgeObservation, FlowForgeState
except ImportError:
    from models import FlowForgeState


class MediumTask:
    """Medium-level task: find an employee and send them an email."""

    task_id = "medium"
    description = "Find an employee in the database and send them an email notification"
    objectives = ["find_employee", "send_email"]

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
        if state.intermediate_data.get("email_sent"):
            completed.append("send_email")
        return completed

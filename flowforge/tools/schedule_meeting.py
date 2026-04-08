"""
Schedule Meeting tool for FlowForge AI.
"""

from typing import Any
try:
    from flowforge.tools.base_tool import BaseTool
except ImportError:
    from tools.base_tool import BaseTool

class ScheduleMeetingTool(BaseTool):
    """Tool to schedule calendar meetings."""

    @property
    def name(self) -> str:
        return "schedule_meeting"

    @property
    def description(self) -> str:
        return "Schedules a calendar meeting given attendees, date, and title."

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "attendees": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of attendee email addresses"
                },
                "date": {
                    "type": "string",
                    "description": "Date and time for the meeting (e.g. '2023-11-20 10:00')"
                },
                "title": {
                    "type": "string",
                    "description": "Meeting title"
                }
            },
            "required": ["attendees", "date", "title"]
        }

    def execute(self, parameters: dict[str, Any]) -> dict[str, Any]:
        attendees = parameters.get("attendees")
        date = parameters.get("date")
        title = parameters.get("title")

        if not attendees or not isinstance(attendees, list):
            return {"success": False, "error": "Valid attendees list is required"}
        
        if not date or not isinstance(date, str):
            return {"success": False, "error": "Valid date string is required"}
            
        if not title or not isinstance(title, str):
            return {"success": False, "error": "Valid title is required"}

        # Simulate scheduling
        return {
            "success": True,
            "data": {
                "meeting_id": "mtg_" + str(hash(title))[-6:],
                "status": "scheduled",
                "attendees": attendees,
                "date": date,
                "title": title
            }
        }

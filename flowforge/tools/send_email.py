"""
FlowForge AI - Send Email Tool

Simulates sending emails without using a real email API.
Validates input fields and returns structured responses.
"""

from __future__ import annotations

import re
from typing import Any

try:
    from flowforge.tools.base_tool import BaseTool
except ImportError:
    from base_tool import BaseTool


class SendEmailTool(BaseTool):
    """Tool for simulating email sending.

    Validates recipient, subject, and body fields before simulating
    the email send operation.  No real email is sent; the operation
    is fully simulated for the FlowForge environment.
    """

    @property
    def name(self) -> str:
        return "send_email"

    @property
    def description(self) -> str:
        return "Send an email to a specified recipient with a subject and body"

    def validate_input(self, parameters: dict[str, Any]) -> tuple[bool, str]:
        """Validate email parameters.

        Checks that 'to', 'subject', and 'body' are present and valid.
        The 'to' field must be a valid email format.
        """
        is_valid, error_msg = super().validate_input(parameters)
        if not is_valid:
            return False, error_msg

        required_fields = ["to", "subject", "body"]
        missing = [f for f in required_fields if f not in parameters]
        if missing:
            return False, f"Missing required fields: {', '.join(missing)}"

        if not isinstance(parameters["to"], str) or not parameters["to"].strip():
            return False, "'to' must be a non-empty string"

        if not isinstance(parameters["subject"], str) or not parameters["subject"].strip():
            return False, "'subject' must be a non-empty string"

        if not isinstance(parameters["body"], str) or not parameters["body"].strip():
            return False, "'body' must be a non-empty string"

        # Basic email format validation
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, parameters["to"].strip()):
            return False, f"Invalid email format: '{parameters['to']}'"

        return True, ""

    def execute(self, parameters: dict[str, Any]) -> dict[str, Any]:
        """Simulate sending an email.

        Args:
            parameters: Dictionary containing 'to', 'subject', and 'body'.

        Returns:
            Structured response with success status and message ID.
        """
        to_addr = parameters["to"].strip()
        subject = parameters["subject"].strip()
        body = parameters["body"].strip()

        # Simulate message ID generation
        message_id = f"msg_{hash(to_addr + subject + body) & 0xFFFFFFFF:08x}"

        return {
            "success": True,
            "data": {
                "message_id": message_id,
                "to": to_addr,
                "subject": subject,
                "status": "sent",
            },
            "error": None,
            "hint": None,
        }

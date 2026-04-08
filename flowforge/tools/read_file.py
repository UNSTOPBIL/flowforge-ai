"""
FlowForge AI - Read File Tool

Simulates reading files from an in-memory file system.
Returns file content with structured error handling.
"""

from __future__ import annotations

from typing import Any

try:
    from flowforge.tools.base_tool import BaseTool
except ImportError:
    from base_tool import BaseTool


class ReadFileTool(BaseTool):
    """Tool for reading files from a simulated in-memory file system.

    Accepts a file_path parameter and returns the file content if found.
    Handles missing files and invalid paths gracefully.
    """

    # Simulated in-memory file system
    _file_system: dict[str, str] = {
        "/reports/sales_summary.txt": "Sales Report\nTotal Sales: $525.00\nRegions: US, EU\nEmployee Count: 42",
        "/reports/complaints_summary.txt": "Complaints Summary\nOpen: 2\nResolved: 0\nEmployees Involved: 5",
        "/reports/analysis.txt": "Regional Analysis\nUnderperforming: EU\nTop Performer: John Smith",
        "/data/customers.csv": "id,name,email,region\n1,Alice,alice@example.com,US\n2,Bob,bob@example.com,EU",
        "/data/sales.csv": "id,product,amount,region,date\n1,Widget A,150.00,US,2026-01-15\n2,Widget B,200.00,EU,2026-01-16",
    }

    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return "Read the content of a file from the simulated file system"

    def validate_input(self, parameters: dict[str, Any]) -> tuple[bool, str]:
        """Validate file reading parameters.

        Checks that 'file_path' is present and is a non-empty string.
        """
        is_valid, error_msg = super().validate_input(parameters)
        if not is_valid:
            return False, error_msg

        if "file_path" not in parameters:
            return False, "Missing required field: 'file_path'"

        if not isinstance(parameters["file_path"], str) or not parameters["file_path"].strip():
            return False, "'file_path' must be a non-empty string"

        return True, ""

    def execute(self, parameters: dict[str, Any]) -> dict[str, Any]:
        """Read a file from the simulated file system.

        Args:
            parameters: Dictionary containing 'file_path'.

        Returns:
            Structured response with file content or error message.
        """
        file_path = parameters["file_path"].strip()

        if file_path not in self._file_system:
            available_files = list(self._file_system.keys())
            return {
                "success": False,
                "data": {},
                "error": f"File not found: '{file_path}'",
                "hint": f"Available files: {', '.join(available_files)}",
            }

        content = self._file_system[file_path]
        return {
            "success": True,
            "data": {
                "file_path": file_path,
                "content": content,
                "size": len(content),
            },
            "error": None,
            "hint": None,
        }

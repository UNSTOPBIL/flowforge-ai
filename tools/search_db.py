"""
FlowForge AI - Search Database Tool

Searches in-memory datasets (employees, services) by query string.
Returns matching records where the query matches name or department.
"""

from __future__ import annotations

from typing import Any

from tools.base_tool import BaseTool


class SearchDBTool(BaseTool):
    """Tool for searching in-memory database records."""

    _employees = [
        {"id": 1, "name": "Alice Johnson", "department": "Engineering"},
        {"id": 2, "name": "Bob Smith", "department": "Marketing"},
        {"id": 3, "name": "Carol White", "department": "Engineering"},
        {"id": 4, "name": "David Brown", "department": "Sales"},
    ]
    _services = [
        {"id": 1, "name": "Cloud Hosting", "status": "active"},
        {"id": 2, "name": "Email Service", "status": "active"},
        {"id": 3, "name": "Analytics", "status": "maintenance"},
    ]

    @property
    def name(self) -> str:
        return "search_db"

    @property
    def description(self) -> str:
        return "Search employees and services databases by keyword"

    def validate_input(self, parameters: dict[str, Any]) -> tuple[bool, str]:
        is_valid, error_msg = super().validate_input(parameters)
        if not is_valid:
            return False, error_msg
        if "query" not in parameters:
            return False, "Missing required field: 'query'"
        if not isinstance(parameters["query"], str):
            return False, "'query' must be a string"
        if not parameters["query"].strip():
            return False, "'query' cannot be empty"
        return True, ""

    def execute(self, parameters: dict[str, Any]) -> dict[str, Any]:
        query = parameters["query"].strip().lower()
        results = []

        for emp in self._employees:
            if query in emp["name"].lower() or query in emp["department"].lower():
                results.append({"type": "employee", **emp})

        for svc in self._services:
            if query in svc["name"].lower() or query in svc["status"].lower():
                results.append({"type": "service", **svc})

        if not results:
            return {
                "success": True,
                "data": [],
                "error": None,
                "hint": "No matching records found. Try a different keyword.",
            }

        return {
            "success": True,
            "data": results,
            "error": None,
            "hint": None,
        }

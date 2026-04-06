"""
FlowForge AI - Run Query Tool

Executes simple SELECT queries against in-memory datasets.
Supports basic filtering with WHERE clauses.
"""

from __future__ import annotations

from typing import Any

from tools.base_tool import BaseTool


class RunQueryTool(BaseTool):
    """Tool for running simple SELECT queries on in-memory data."""

    _data = {
        "employees": [
            {"id": 1, "name": "Alice", "department": "Engineering", "salary": 90000},
            {"id": 2, "name": "Bob", "department": "Marketing", "salary": 70000},
            {"id": 3, "name": "Carol", "department": "Engineering", "salary": 95000},
            {"id": 4, "name": "David", "department": "Sales", "salary": 65000},
        ],
        "services": [
            {"id": 1, "name": "Cloud Hosting", "status": "active"},
            {"id": 2, "name": "Email Service", "status": "active"},
            {"id": 3, "name": "Analytics", "status": "maintenance"},
        ],
    }

    @property
    def name(self) -> str:
        return "run_query"

    @property
    def description(self) -> str:
        return "Run simple SELECT queries on employees and services tables"

    def validate_input(self, parameters: dict[str, Any]) -> tuple[bool, str]:
        is_valid, error_msg = super().validate_input(parameters)
        if not is_valid:
            return False, error_msg
        if "query" not in parameters:
            return False, "Missing required field: 'query'"
        if not isinstance(parameters["query"], str) or not parameters["query"].strip():
            return False, "'query' must be a non-empty string"
        if not parameters["query"].strip().upper().startswith("SELECT"):
            return False, "Only SELECT queries are allowed"
        return True, ""

    def execute(self, parameters: dict[str, Any]) -> dict[str, Any]:
        query = parameters["query"].strip()
        parts = query.lower().split()

        if len(parts) < 4 or parts[2] != "from":
            return {"success": False, "data": [], "error": "Invalid query format. Use: SELECT * FROM <table>", "hint": "Example: SELECT * FROM employees"}

        table = parts[3]
        if table not in self._data:
            return {"success": False, "data": [], "error": f"Unknown table: '{table}'", "hint": f"Available tables: {', '.join(self._data.keys())}"}

        results = list(self._data[table])

        # Simple WHERE clause support
        if "where" in parts:
            where_idx = parts.index("where")
            if where_idx + 3 <= len(parts):
                field = parts[where_idx + 1]
                value = parts[where_idx + 3].strip("'\"")
                results = [r for r in results if str(r.get(field, "")).lower() == value.lower()]

        return {"success": True, "data": results, "error": None, "hint": None}

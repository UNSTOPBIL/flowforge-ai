"""
FlowForge AI - Base Tool

Abstract base class for all tools in the FlowForge environment.
Provides a standardized interface for tool execution, input validation,
and structured error handling.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseTool(ABC):
    """Abstract base class for all FlowForge tools.

    All concrete tools must inherit from this class and implement the
    required properties and methods.  This ensures a consistent interface
    for the environment to call any tool without knowing its specifics.

    Subclasses must implement:
        - name: A short identifier for the tool (e.g. "database")
        - description: A human-readable description of what the tool does
        - execute: The core logic that processes parameters and returns results

    Example usage:
        class DatabaseTool(BaseTool):
            @property
            def name(self) -> str:
                return "database"

            @property
            def description(self) -> str:
                return "Query and manipulate database tables"

            def execute(self, parameters: dict) -> dict:
                # Tool-specific logic here
                return {"success": True, "data": {...}, "error": None, "hint": None}
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the unique identifier for this tool.

        Must be a lowercase string with no spaces (e.g. "database", "email").
        Used by the environment to route actions to the correct tool.
        """
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Return a human-readable description of this tool.

        Used in prompts to help the AI agent understand what the tool does
        and when to use it.
        """
        pass

    @abstractmethod
    def execute(self, parameters: dict[str, Any]) -> dict[str, Any]:
        """Execute the tool with the given parameters.

        Args:
            parameters: A dictionary of parameters specific to the tool
                        and action being performed.

        Returns:
            A structured dictionary with the following keys:
                - success (bool): Whether the tool execution succeeded
                - data (dict): The result data (empty dict on failure)
                - error (str | None): Error message if execution failed
                - hint (str | None): Helpful hint for the agent on how to fix the issue

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        pass

    def validate_input(self, parameters: dict[str, Any]) -> tuple[bool, str]:
        """Validate input parameters before execution.

        Override this method in subclasses to implement tool-specific
        validation logic.  The default implementation checks that
        parameters is a non-empty dictionary.

        Args:
            parameters: The parameters dictionary to validate.

        Returns:
            A tuple of (is_valid, error_message).  If is_valid is True,
            error_message will be an empty string.
        """
        if not isinstance(parameters, dict):
            return False, "parameters must be a dictionary"
        if not parameters:
            return False, "parameters cannot be empty"
        return True, ""

    def safe_execute(self, parameters: dict[str, Any]) -> dict[str, Any]:
        """Execute the tool with built-in error handling.

        This method wraps execute() to catch any unexpected exceptions
        and return a structured error response instead of crashing.

        Args:
            parameters: The parameters to pass to execute().

        Returns:
            A structured response dict.  If execution succeeds, returns
            the result from execute().  If an exception occurs, returns
            an error response.
        """
        # Validate input first
        is_valid, error_msg = self.validate_input(parameters)
        if not is_valid:
            return {
                "success": False,
                "data": {},
                "error": error_msg,
                "hint": f"Check the required parameters for the '{self.name}' tool",
            }

        # Execute with exception handling
        try:
            result = self.execute(parameters)
            # Ensure result has the expected structure
            if not isinstance(result, dict):
                return {
                    "success": False,
                    "data": {},
                    "error": "Tool returned invalid response format",
                    "hint": f"The '{self.name}' tool must return a dictionary",
                }
            # Ensure all required keys are present
            for key in ("success", "data", "error", "hint"):
                if key not in result:
                    result[key] = None if key in ("error", "hint") else ({} if key == "data" else False)
            return result
        except Exception as e:
            return {
                "success": False,
                "data": {},
                "error": f"Unexpected error: {str(e)}",
                "hint": f"An unexpected error occurred in the '{self.name}' tool. Try again with different parameters.",
            }

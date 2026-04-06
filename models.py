"""
FlowForge AI - Data Models

Structured data models for actions, observations, and environment state.
Lightweight dataclass-based implementation with built-in validation and
serialization support. No external heavy dependencies required.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class FlowForgeAction:
    """Represents a single structured action taken by the AI agent.

    Actions are JSON-serializable instructions that specify which tool to
    invoke and with what parameters.  The environment validates and
    executes these actions inside step().

    Attributes:
        tool_name: Name of the tool to invoke (e.g. "database", "email",
                   "filesystem", "query").  Must be a non-empty string.
        parameters: Keyword-argument dictionary passed to the tool.
                    Defaults to an empty dict when omitted.
    """

    tool_name: str
    parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate fields immediately after instantiation."""
        if not isinstance(self.tool_name, str) or not self.tool_name.strip():
            raise ValueError("tool_name must be a non-empty string")
        if not isinstance(self.parameters, dict):
            raise TypeError("parameters must be a dict")
        # Normalise
        self.tool_name = self.tool_name.strip().lower()

    # -- Serialization -------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dictionary representation."""
        return {"tool_name": self.tool_name, "parameters": dict(self.parameters)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FlowForgeAction:
        """Construct a FlowForgeAction from a dictionary.

        Raises ValueError for missing required keys or invalid types.
        """
        if not isinstance(data, dict):
            raise TypeError("data must be a dict")
        if "tool_name" not in data:
            raise ValueError("missing required key: 'tool_name'")
        return cls(
            tool_name=data["tool_name"],
            parameters=data.get("parameters", {}),
        )

    def __repr__(self) -> str:
        return f"FlowForgeAction(tool_name='{self.tool_name}', parameters={self.parameters!r})"


@dataclass
class FlowForgeObservation:
    """Represents the environment's response after executing an action.

    Observations are returned by step() and fed back to the agent for the
    next reasoning cycle.

    Attributes:
        message: Human-readable description of the result or error.
        available_tools: List of tool names the agent may call next.
        state_summary: Lightweight snapshot of the current environment
                       state (e.g. progress indicators, flags).  Defaults
                       to an empty dict.
    """

    message: str
    available_tools: list[str]
    state_summary: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate fields immediately after instantiation."""
        if not isinstance(self.message, str):
            raise TypeError("message must be a string")
        if not isinstance(self.available_tools, list):
            raise TypeError("available_tools must be a list")
        if not isinstance(self.state_summary, dict):
            raise TypeError("state_summary must be a dict")

    # -- Serialization -------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dictionary representation."""
        return {
            "message": self.message,
            "available_tools": list(self.available_tools),
            "state_summary": dict(self.state_summary),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FlowForgeObservation:
        """Construct a FlowForgeObservation from a dictionary.

        Raises ValueError for missing required keys or invalid types.
        """
        if not isinstance(data, dict):
            raise TypeError("data must be a dict")
        missing = [k for k in ("message", "available_tools") if k not in data]
        if missing:
            raise ValueError(f"missing required key(s): {', '.join(missing)}")
        return cls(
            message=data["message"],
            available_tools=data["available_tools"],
            state_summary=data.get("state_summary", {}),
        )

    def __repr__(self) -> str:
        return (f"FlowForgeObservation(message='{self.message[:50]}...', "
                f"available_tools={self.available_tools!r})")


@dataclass
class FlowForgeState:
    """Tracks the full internal state of an episode.

    The environment maintains one instance per episode and updates it on
    every step().  This class is deliberately free of business logic —
    it is a pure data container with safe defaults and deep-copy
    serialisation.

    Attributes:
        completed_objectives: Ordered list of objective identifiers that
                              the agent has successfully completed.
        step_count: Number of steps taken so far in the current episode.
        max_steps: Upper bound on steps before forced termination.
                  Must be non-negative (zero means episode is done immediately).
        task_id: Unique identifier of the active task.
        intermediate_data: Arbitrary key-value store for task-specific
                           progress (e.g. files created, queries run).
                           Defaults to an empty dict.
    """

    completed_objectives: list[str] = field(default_factory=list)
    step_count: int = 0
    max_steps: int = 20
    task_id: str = ""
    intermediate_data: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate fields immediately after instantiation."""
        if not isinstance(self.completed_objectives, list):
            raise TypeError("completed_objectives must be a list")
        if not isinstance(self.step_count, int) or self.step_count < 0:
            raise ValueError("step_count must be a non-negative integer")
        if not isinstance(self.max_steps, int) or self.max_steps < 0:
            raise ValueError("max_steps must be a non-negative integer")
        if not isinstance(self.task_id, str):
            raise TypeError("task_id must be a string")
        if not isinstance(self.intermediate_data, dict):
            raise TypeError("intermediate_data must be a dict")

    # -- Convenience helpers -------------------------------------------------

    @property
    def is_done(self) -> bool:
        """Return True when the step limit has been reached."""
        return self.step_count >= self.max_steps

    @property
    def progress(self) -> float:
        """Return completion ratio capped at 1.0."""
        if self.max_steps == 0:
            return 0.0
        return min(1.0, self.step_count / self.max_steps)

    # -- Serialization -------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dictionary representation (deep copy)."""
        return {
            "completed_objectives": list(self.completed_objectives),
            "step_count": self.step_count,
            "max_steps": self.max_steps,
            "task_id": self.task_id,
            "intermediate_data": copy.deepcopy(self.intermediate_data),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FlowForgeState:
        """Construct a FlowForgeState from a dictionary.

        Missing optional keys fall back to safe defaults.
        """
        if not isinstance(data, dict):
            raise TypeError("data must be a dict")
        return cls(
            completed_objectives=data.get("completed_objectives", []),
            step_count=data.get("step_count", 0),
            max_steps=data.get("max_steps", 20),
            task_id=data.get("task_id", ""),
            intermediate_data=data.get("intermediate_data", {}),
        )

    def __repr__(self) -> str:
        return (f"FlowForgeState(step_count={self.step_count}/{self.max_steps}, "
                f"task_id='{self.task_id}', objectives={len(self.completed_objectives)})")
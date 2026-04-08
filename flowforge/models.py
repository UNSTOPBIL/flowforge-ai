"""
FlowForge AI - Data Models

Structured data models for actions, observations, and environment state.
Uses Pydantic BaseModel for validation and JSON serialization support.
"""

from __future__ import annotations

import copy
from typing import Any
from pydantic import BaseModel, Field, field_validator


class FlowForgeAction(BaseModel):
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
    parameters: dict[str, Any] = Field(default_factory=dict)

    @field_validator('tool_name')
    @classmethod
    def validate_tool_name(cls, v: str) -> str:
        """Validate tool_name."""
        VALID_TOOLS = {"search_db", "send_email", "read_file", "run_query", "schedule_meeting", "finish"}
        if not v or not v.strip():
            raise ValueError("tool_name must be a non-empty string")
        v = v.strip().lower()
        if v not in VALID_TOOLS:
            raise ValueError(f"Unknown tool: {v!r}")
        return v

    @field_validator('parameters')
    @classmethod
    def validate_parameters(cls, v: dict[str, Any]) -> dict[str, Any]:
        """Validate parameters."""
        if not isinstance(v, dict):
            raise TypeError("parameters must be a dict")
        return v

    def __repr__(self) -> str:
        return f"FlowForgeAction(tool_name='{self.tool_name}', parameters={self.parameters!r})"


class FlowForgeObservation(BaseModel):
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
    state_summary: dict[str, Any] = Field(default_factory=dict)

    @field_validator('message')
    @classmethod
    def validate_message(cls, v: str) -> str:
        """Validate message."""
        if not isinstance(v, str):
            raise TypeError("message must be a string")
        return v

    @field_validator('available_tools')
    @classmethod
    def validate_available_tools(cls, v: list[str]) -> list[str]:
        """Validate available_tools."""
        if not isinstance(v, list):
            raise TypeError("available_tools must be a list")
        return v

    @field_validator('state_summary')
    @classmethod
    def validate_state_summary(cls, v: dict[str, Any]) -> dict[str, Any]:
        """Validate state_summary."""
        if not isinstance(v, dict):
            raise TypeError("state_summary must be a dict")
        return v

    def __repr__(self) -> str:
        return (f"FlowForgeObservation(message='{self.message[:50]}...', "
                f"available_tools={self.available_tools!r})")


class FlowForgeState(BaseModel):
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

    completed_objectives: list[str] = Field(default_factory=list)
    step_count: int = 0
    max_steps: int = 20
    task_id: str = ""
    intermediate_data: dict[str, Any] = Field(default_factory=dict)
    
    # RL Tracking Fields
    last_action: str = ""
    last_result_summary: str = ""
    action_history: list[str] = Field(default_factory=list)

    @field_validator('completed_objectives')
    @classmethod
    def validate_completed_objectives(cls, v: list[str]) -> list[str]:
        """Validate completed_objectives."""
        if not isinstance(v, list):
            raise TypeError("completed_objectives must be a list")
        return v

    @field_validator('step_count')
    @classmethod
    def validate_step_count(cls, v: int) -> int:
        """Validate step_count."""
        if not isinstance(v, int) or v < 0:
            raise ValueError("step_count must be a non-negative integer")
        return v
        
    @field_validator('action_history')
    @classmethod
    def validate_action_history(cls, v: list[str]) -> list[str]:
        """Validate action_history."""
        if not isinstance(v, list):
            raise TypeError("action_history must be a list")
        return v

    @field_validator('max_steps')
    @classmethod
    def validate_max_steps(cls, v: int) -> int:
        """Validate max_steps."""
        if not isinstance(v, int) or v < 0:
            raise ValueError("max_steps must be a non-negative integer")
        return v

    @field_validator('task_id')
    @classmethod
    def validate_task_id(cls, v: str) -> str:
        """Validate task_id."""
        if not isinstance(v, str):
            raise TypeError("task_id must be a string")
        return v

    @field_validator('intermediate_data')
    @classmethod
    def validate_intermediate_data(cls, v: dict[str, Any]) -> dict[str, Any]:
        """Validate intermediate_data."""
        if not isinstance(v, dict):
            raise TypeError("intermediate_data must be a dict")
        return v

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

    def __repr__(self) -> str:
        return (f"FlowForgeState(step_count={self.step_count}/{self.max_steps}, "
                f"task_id='{self.task_id}', objectives={len(self.completed_objectives)})")
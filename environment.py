"""
FlowForge AI - Environment

OpenEnv-compatible environment for AI workflow automation.
Manages tool execution, state tracking, and reward computation.
"""

from __future__ import annotations

from typing import Any

from models import FlowForgeAction, FlowForgeObservation, FlowForgeState
from tools.search_db import SearchDBTool
from tools.send_email import SendEmailTool
from tools.read_file import ReadFileTool
from tools.run_query import RunQueryTool


class FlowForgeEnvironment:
    """OpenEnv environment for FlowForge AI workflow tasks.

    Manages a collection of tools, tracks episode state, and computes
    rewards based on agent actions and progress toward objectives.
    """

    def __init__(self) -> None:
        self._tools = {
            "search_db": SearchDBTool(),
            "send_email": SendEmailTool(),
            "read_file": ReadFileTool(),
            "run_query": RunQueryTool(),
        }
        self._state: FlowForgeState | None = None

    def reset(self, seed=None, episode_id=None, **kwargs) -> FlowForgeObservation:
        """Reset the environment to initial state for a new episode.

        Args:
            seed: Random seed for reproducibility (unused in deterministic mode).
            episode_id: Unique identifier for the episode.
            **kwargs: Additional configuration (task_id, max_steps).

        Returns:
            Initial observation with available tools and empty state.
        """
        self._state = FlowForgeState(
            step_count=0,
            max_steps=kwargs.get("max_steps", 20),
            task_id=kwargs.get("task_id", "default"),
            completed_objectives=[],
        )
        return FlowForgeObservation(
            message="Environment reset. Ready for actions.",
            available_tools=list(self._tools.keys()),
            state_summary=self._get_state_summary(),
        )

    def step(self, action, timeout_s=None, **kwargs) -> tuple[FlowForgeObservation, float, bool, dict]:
        """Execute one environment step.

        Args:
            action: Dict with 'tool_name' and 'parameters', or FlowForgeAction.
            timeout_s: Optional timeout in seconds (unused in simulation).
            **kwargs: Additional step options.

        Returns:
            Tuple of (observation, reward, done, info).
        """
        if self._state is None:
            self.reset()

        self._state.step_count += 1
        reward = 0.0
        done = False
        message = ""

        try:
            # Parse and validate action
            if isinstance(action, dict):
                action = FlowForgeAction.from_dict(action)
            elif not isinstance(action, FlowForgeAction):
                raise ValueError("Action must be a dict or FlowForgeAction")

            # Check tool exists
            tool = self._tools.get(action.tool_name)
            if not tool:
                reward = -0.1
                message = f"Unknown tool: '{action.tool_name}'"
            else:
                # Execute tool
                result = tool.safe_execute(action.parameters)
                if result["success"]:
                    reward = 0.2  # Valid execution
                    # Check for meaningful progress
                    if self._check_progress(action.tool_name, result):
                        reward += 0.2
                    message = f"Tool '{action.tool_name}' executed successfully"
                else:
                    reward = -0.1
                    message = f"Tool error: {result['error']}"
        except Exception as e:
            reward = -0.1
            message = f"Invalid action: {str(e)}"

        # Check completion
        done = self._state.is_done or self._check_all_objectives_done()
        if done:
            message += " [Episode complete]"

        obs = FlowForgeObservation(
            message=message,
            available_tools=list(self._tools.keys()),
            state_summary=self._get_state_summary(),
        )
        info = {"step": self._state.step_count, "reward": reward}
        return obs, reward, done, info

    def _check_progress(self, tool_name: str, result: dict) -> bool:
        """Check if tool execution represents meaningful progress."""
        if not result.get("data"):
            return False
        if tool_name == "search_db" and len(result["data"]) > 0:
            self._state.completed_objectives.append(f"search_{tool_name}")
            return True
        if tool_name == "send_email":
            self._state.completed_objectives.append("send_email")
            return True
        if tool_name == "read_file" and result["data"].get("content"):
            self._state.completed_objectives.append("read_file")
            return True
        if tool_name == "run_query" and len(result["data"]) > 0:
            self._state.completed_objectives.append("run_query")
            return True
        return False

    def _check_all_objectives_done(self) -> bool:
        """Check if all required objectives are completed."""
        required = {"search_db", "send_email", "read_file", "run_query"}
        completed = {obj for obj in self._state.completed_objectives if any(req in obj for req in required)}
        return len(completed) >= len(required)

    def _get_state_summary(self) -> dict[str, Any]:
        """Return a lightweight summary of current state."""
        return {
            "step_count": self._state.step_count,
            "max_steps": self._state.max_steps,
            "completed_objectives": list(self._state.completed_objectives),
            "progress": self._state.progress,
            "done": self._state.is_done,
        }

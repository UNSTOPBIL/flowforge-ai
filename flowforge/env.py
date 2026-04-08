"""
FlowForge AI - Environment

OpenEnv-compatible environment for AI workflow automation.
Manages tool execution, state tracking, and reward computation.
"""

from __future__ import annotations

from typing import Any

try:
    from flowforge.models import FlowForgeAction, FlowForgeObservation, FlowForgeState
    from flowforge.tools.search_db import SearchDBTool
    from flowforge.tools.send_email import SendEmailTool
    from flowforge.tools.read_file import ReadFileTool
    from flowforge.tools.run_query import RunQueryTool
    from flowforge.tools.schedule_meeting import ScheduleMeetingTool
except ImportError:
    from models import FlowForgeAction, FlowForgeObservation, FlowForgeState
    from tools.search_db import SearchDBTool
    from tools.send_email import SendEmailTool
    from tools.read_file import ReadFileTool
    from tools.run_query import RunQueryTool
    from tools.schedule_meeting import ScheduleMeetingTool


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
            "schedule_meeting": ScheduleMeetingTool(),
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
            intermediate_data={},  # Initialize intermediate_data
            last_action="",
            last_result_summary="",
            action_history=[],
        )
        return FlowForgeObservation(
            message=f"Environment reset. Task: {self._get_task_description()}. "
                    f"Objectives: {', '.join(self._get_task_objectives())}. Ready for actions.",
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
                action = FlowForgeAction.model_validate(action)
            elif not isinstance(action, FlowForgeAction):
                raise ValueError("Action must be a dict or FlowForgeAction")

            if action.tool_name == "finish":
                done = True
                message = "Task explicitly finished by agent"
                base_finish_reward = 0.1
                completed = self._state.completed_objectives
                required = set(self._get_task_objectives())
                if required and required.issubset(set(completed)):
                    efficiency_bonus = max(0.0, 0.5 * (1 - self._state.step_count / max(1, self._state.max_steps)))
                else:
                    efficiency_bonus = 0.0
                reward += base_finish_reward + efficiency_bonus
                
            else:
                # Check tool exists
                tool = self._tools.get(action.tool_name)
                if not tool:
                    reward -= 0.2
                    message = f"Unknown tool: '{action.tool_name}'"
                else:
                    # Execute tool
                    tool_count = self._state.action_history.count(action.tool_name)
                    if tool_count > 0:
                        loop_penalty = min(0.2, 0.05 * tool_count)
                        reward -= loop_penalty

                    # Task-aware base reward shaping
                    TOOL_RELEVANCE = {
                        "easy":   {"search_db": 1.0, "send_email": 0.5, "read_file": 0.5, "run_query": 0.5, "schedule_meeting": 0.5},
                        "medium": {"search_db": 1.0, "send_email": 1.0, "read_file": 0.3, "run_query": 0.3, "schedule_meeting": 0.5},
                        "hard":   {"read_file": 1.0, "run_query": 1.0, "schedule_meeting": 1.0, "send_email": 1.0, "search_db": 0.3},
                    }
                    relevance = TOOL_RELEVANCE.get(self._state.task_id, {}).get(action.tool_name, 0.5)

                    result = tool.safe_execute(action.parameters)
                    self._state.last_action = action.tool_name
                    self._state.action_history.append(action.tool_name)
                    self._state.last_result_summary = str(result.get("data", ""))[:100]

                    if result["success"]:
                        reward += 0.2 * relevance  # Valid execution
                        # Check for meaningful progress and update state
                        progress_made, objective_updates = self._check_progress(action.tool_name, result)
                        
                        new_objective_found = False
                        for obj in objective_updates.get("objectives", []):
                            if obj not in self._state.completed_objectives:
                                self._state.completed_objectives.append(obj)
                                new_objective_found = True
                        
                        if new_objective_found:
                            task_objs = self._get_task_objectives()
                            completion_ratio = len(set(self._state.completed_objectives) & set(task_objs)) / max(len(task_objs), 1)
                            reward += 0.3 * (1.0 + completion_ratio)
                        elif progress_made:
                            reward += 0.1  # Sub-goal proximity
                            
                        self._state.intermediate_data.update(objective_updates.get("intermediate", {}))
                        
                        msg_str = str(result.get("data", ""))[:100]
                        if result.get("data") and isinstance(result["data"], dict) and "status" in result["data"]:
                            msg_str = f"Status: {result['data']['status']}"
                        elif result.get("data") and isinstance(result["data"], list) and len(result["data"]) > 0:
                            msg_str = f"Found {len(result['data'])} records"
                        elif result.get("data") and isinstance(result["data"], dict) and "success" in result["data"]:
                            msg_str = "Operation complete"
                            
                        message = f"Tool '{action.tool_name}' executed successfully. Result: {msg_str}"
                    else:
                        reward -= 0.1
                        message = f"Tool error: {result['error']}"
        except Exception as e:
            reward = -0.1
            message = f"Invalid action: {str(e)}"

        # Check completion
        done = done or self._state.is_done or self._check_all_objectives_done()
        if done:
            message += " [Episode complete]"

        obs = FlowForgeObservation(
            message=message,
            available_tools=list(self._tools.keys()),
            state_summary=self._get_state_summary(),
        )
        info = {"step": self._state.step_count, "reward": reward}
        return obs, reward, done, info

    def _check_progress(self, tool_name: str, result: dict) -> tuple[bool, dict]:
        """Check if tool execution represents meaningful progress and return updates.

        Returns:
            Tuple of (progress_made: bool, updates: dict)
            updates contains:
                - intermediate: dict for intermediate_data updates
                - objectives: list for completed_objectives updates
        """
        intermediate_updates = {}
        objective_updates = []
        
        if not result.get("data"):
            return False, {"intermediate": intermediate_updates, "objectives": objective_updates}
            
        if tool_name == "search_db" and len(result["data"]) > 0:
            # Found employee data
            intermediate_updates["employee_found"] = True
            intermediate_updates["employee_data_read"] = True  # Assume search returns employee data
            objective_updates.append("find_employee")
            objective_updates.append("read_employee_data")
            return True, {"intermediate": intermediate_updates, "objectives": objective_updates}
        if tool_name == "send_email":
            # Email sent successfully
            intermediate_updates["email_sent"] = True
            objective_updates.append("send_email")
            return True, {"intermediate": intermediate_updates, "objectives": objective_updates}
        if tool_name == "read_file" and result["data"].get("content"):
            # File read successfully
            intermediate_updates["file_read"] = True
            # Check if this is employee data
            content = result["data"].get("content", "")
            if "employee" in content.lower() or "name" in content.lower() or "id" in content.lower():
                intermediate_updates["employee_data_read"] = True
                objective_updates.append("read_employee_data")
            objective_updates.append("read_file")
            return True, {"intermediate": intermediate_updates, "objectives": objective_updates}
        if tool_name == "run_query" and len(result["data"]) > 0:
            # Query executed successfully
            intermediate_updates["query_done"] = True
            objective_updates.append("run_query")
            return True, {"intermediate": intermediate_updates, "objectives": objective_updates}
        if tool_name == "schedule_meeting":
            # Meeting scheduled successfully
            intermediate_updates["meeting_scheduled"] = True
            objective_updates.append("schedule_meeting")
            return True, {"intermediate": intermediate_updates, "objectives": objective_updates}
        return False, {"intermediate": intermediate_updates, "objectives": objective_updates}

    def _check_all_objectives_done(self) -> bool:
        """Check if all required objectives for the current task are completed.

        Uses the actual objective names rather than just a count, preventing
        false positives when unrelated objectives are completed.
        """
        task_required_objectives = {
            "easy": {"find_employee", "read_employee_data"},
            "medium": {"find_employee", "send_email"},
            "hard": {"read_file", "run_query", "schedule_meeting", "send_email"},
        }
        required = task_required_objectives.get(self._state.task_id)
        if not required:
            return False
        completed = set(self._state.completed_objectives)
        return required.issubset(completed)

    def _get_task_objectives(self) -> list[str]:
        task_required_objectives = {
            "easy": ["find_employee", "read_employee_data"],
            "medium": ["find_employee", "send_email"],
            "hard": ["read_file", "run_query", "schedule_meeting", "send_email"],
        }
        return task_required_objectives.get(self._state.task_id, [])

    def _get_task_description(self) -> str:
        task_descriptions = {
            "easy": "Search the database to find an employee by name or department",
            "medium": "Find an employee in the database and send them an email notification",
            "hard": "Read a report file, run a database query for analysis, schedule a review meeting, and send the results via email"
        }
        return task_descriptions.get(self._state.task_id, "Unknown task")

    def _get_task_difficulty(self) -> str:
        return self._state.task_id if self._state.task_id in ["easy", "medium", "hard"] else "unknown"

    def _get_state_summary(self) -> dict[str, Any]:
        """Return an enriched summary of current state with RL-critical signals."""
        objective_progress = {}
        task_objectives = self._get_task_objectives()
        for obj in task_objectives:
            objective_progress[obj] = 1.0 if obj in self._state.completed_objectives else 0.0

        return {
            "step_count": self._state.step_count,
            "max_steps": self._state.max_steps,
            "task_id": self._state.task_id,
            "completed_objectives": list(self._state.completed_objectives),
            "intermediate_data": dict(self._state.intermediate_data),
            "progress": self._state.progress,
            "done": self._state.is_done,
            "task_description": self._get_task_description(),
            "required_objectives": task_objectives,
            "objective_progress": objective_progress,
            "completion_ratio": len(set(self._state.completed_objectives) & set(task_objectives)) / max(len(task_objectives), 1) if task_objectives else 0.0,
            "remaining_steps": max(0, self._state.max_steps - self._state.step_count),
            "last_action": self._state.last_action,
            "last_tool_outcome": str(self._state.last_result_summary),
            "tool_call_count": {tool: self._state.action_history.count(tool) for tool in set(self._state.action_history)},
            "last_result_summary": self._state.last_result_summary,
            "action_history": list(self._state.action_history),
            "difficulty": self._get_task_difficulty(),
        }

    def state(self) -> FlowForgeState:
        """Return current environment state."""
        if self._state is None:
             self.reset()
        return self._state

    def close(self):
        """Clean up the environment resources."""
        self._state = None

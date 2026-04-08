---
title: FlowForge AI
emoji: 🔧
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
---
# FlowForge AI

[![OpenEnv Compatible](https://img.shields.io/badge/OpenEnv-Compatible-brightgreen)](https://huggingface.co/spaces/open-env/validator)
[![Meta PyTorch Hackathon](https://img.shields.io/badge/Meta%20PyTorch-Hackathon%202024-blue)](https://huggingface.co/open-env)

**FlowForge AI** is an OpenEnv-compatible Reinforcement Learning environment where LLM agents learn to automate real-world enterprise workflows.

## Motivation & Description

Instead of a simple game or toy logic problem, FlowForge simulates actual back-office operations. Agents act as automated HR/operations assistants capable of reading files, searching employee databases, running SQL queries, scheduling meetings, and sending emails — in the right sequence — to complete tasks. This models genuine enterprise environments where agents must:
- Plan multi-step tool sequences
- Recover from errors and invalid actions
- Synthesize information across multiple sources
- Avoid redundant actions (anti-loop reward shaping)

## Action Space

The Action Space is strictly defined via the Pydantic `FlowForgeAction` model. Each action is a JSON payload:
```json
{
  "tool_name": "search_db",
  "parameters": {"query": "engineering"}
}
```

**Available Tools:**
| Tool | Parameters | Description |
|------|-----------|-------------|
| `search_db` | `query: str` | Search employee/service database |
| `send_email` | `to: str, subject: str, body: str` | Send simulated email |
| `read_file` | `file_path: str` | Read internal reports/files |
| `run_query` | `query: str` | Execute SQL SELECT on database |
| `schedule_meeting` | `attendees: list, date: str, title: str` | Schedule a calendar meeting |
| `finish` | _(none)_ | Signal task completion |

All `tool_name` values are validated against this whitelist — unknown tools are rejected and penalized.

## Observation Space

Defined via the `FlowForgeObservation` Pydantic model:
| Field | Type | Description |
|-------|------|-------------|
| `message` | `str` | Free-text feedback from the environment |
| `data` | `dict` | Structured result data (query rows, file contents, etc.) |
| `error` | `bool` | True if the previous action failed |
| `available_tools` | `list[str]` | Tools available in the current episode |
| `state_summary` | `dict` | Step count, progress, action history, tool usage stats |

## Tasks & Difficulty Progression

FlowForge AI provides **3 strictly graded tasks**:

1. **Easy** (`task_easy.py`): Search the database to find an employee by name or department. Tests basic tool usage and parameter filling.
2. **Medium** (`task_medium.py`): Find an employee in the database and send them an email notification. Tests sequencing logic and data transfer across tools.
3. **Hard** (`task_hard.py`): Read a report file, run a database query, schedule a review meeting, then send results via email. Tests complex multi-step planning, tool sequencing, and cross-tool data synthesis.

## Reward Function

The reward function provides **dense, task-aware signals** over the trajectory rather than sparse binary rewards:

| Signal | Value | Condition |
|--------|-------|-----------|
| Tool execution bonus | `+0.2 × relevance` | Successful tool call (scaled by task-tool relevance) |
| Objective progress bonus | `+0.3 × (1 + completion_ratio)` | First time a new objective is satisfied |
| Sub-goal proximity | `+0.1` | Intermediate progress without completing a new objective |
| Finish reward | `+0.1` | Clean termination via `finish` |
| Tool failure | `-0.1` | Invalid parameters or execution error |
| Unknown tool | `-0.2` | Attempting a tool that doesn't exist |
| Loop penalty | `-0.05 × frequency` | Repeating the same tool (capped at -0.2) |

**Task-Aware Relevance Multipliers:**
- Easy: `search_db=1.0`, others `0.5`
- Medium: `search_db=1.0, send_email=1.0`, others `0.3–0.5`
- Hard: `read_file=1.0, run_query=1.0, schedule_meeting=1.0, send_email=1.0`, `search_db=0.3`

## Baseline Scores

Testing the deterministic rule-based `inference.py` yields:
| Task | Score |
|------|-------|
| Easy | **1.0** |
| Medium | **1.0** |
| Hard | **1.0** |
| **Average** | **1.0** |

When evaluated against LLMs (GPT-4o, Llama-3-70B), scores depend on the model's tool-sequencing reliability and ability to avoid loops.

## Setup & Usage Instructions

### Local Execution
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run the deterministic baseline (all 3 tasks)
python inference.py

# Run judge simulation dashboard
python judge_sim.py

# Run interactive RL reward explorer
python interactive_sim.py
```

### Run Full Test Suite
```bash
export PYTHONPATH=.
python -m pytest tests/ -v
# Expected: 105 passed ✅
```

### Docker
```bash
docker build -t flowforge-ai .
docker run -p 7860:7860 --cpus=2 --memory=8g flowforge-ai
```

### Validation
To run the strict Meta PyTorch Hackathon checklist:
```bash
chmod +x validate-submission.sh
./validate-submission.sh
```

## Repository Structure
```
FlowForge/
├── inference.py              # Main entry point (OpenEnv required)
├── openenv.yaml              # Environment metadata
├── Dockerfile                # Docker container spec
├── requirements.txt          # Python dependencies
├── validate-submission.sh    # Pre-submission validation script
├── judge_sim.py              # Live judge evaluation dashboard
├── interactive_sim.py        # Interactive RL reward explorer
├── flowforge/
│   ├── env.py                # FlowForgeEnvironment (OpenEnv API)
│   ├── models.py             # Pydantic action/observation/state models
│   ├── grader.py             # Objective-based scoring
│   ├── tasks/
│   │   ├── task_easy.py
│   │   ├── task_medium.py
│   │   └── task_hard.py
│   └── tools/
│       ├── base_tool.py      # Abstract BaseTool
│       ├── search_db.py
│       ├── send_email.py
│       ├── read_file.py
│       ├── run_query.py
│       └── schedule_meeting.py
└── tests/                    # 105 E2E tests
```

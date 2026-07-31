<div align="center">

# FlowForge AI

**An OpenEnv-compatible Reinforcement Learning environment for Enterprise Workflow Automation.**

[![OpenEnv Compatible](https://img.shields.io/badge/OpenEnv-Compatible-brightgreen)](https://huggingface.co/spaces/open-env/validator)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

FlowForge simulates actual back-office operations where LLM agents act as automated HR/operations assistants, learning to synthesize information, manage tools, and recover from real-world errors.

</div>

---

## Key Features
- **Genuine Enterprise Operations**: Move beyond toy environments. Agents read files, search employee databases, run SQL queries, schedule meetings, and send emails.
- **Strictly Defined Action Space**: Validated entirely via Pydantic -- preventing hallucinatory tool calls.
- **Task-Aware Reward Shaping**: Dense reward signals that adapt based on the task (e.g., `read_file` is crucial for hard tasks, but optional for easy ones).
- **Anti-Loop Architecture**: Punishes infinite loops and duplicate actions to teach agents efficient planning.
- **Zero-Cost Baseline**: Run locally and test deterministically without eating up OpenAI credits.

---

## How it Works
```mermaid
graph TD
    A[LLM Agent] -->|Action JSON| B(FlowForge Environment)
    B -->|Validation| C{Valid Tool?}
    C -- No --> D[Error Observation + Negative Reward]
    C -- Yes --> E[Execute Tool]
    E --> F(State Tracker)
    F -->|Objective Check| G{Task Complete?}
    G -- Yes --> H[Success Observation + Finish Reward]
    G -- No --> I[Result Observation + Progress Reward]
    D --> A
    I --> A
    H --> J((Episode End))
    
    classDef default fill:#1f2937,stroke:#3b82f6,stroke-width:2px,color:#f3f4f6;
    classDef logic fill:#374151,stroke:#f59e0b,stroke-width:2px,color:#f3f4f6;
    classDef success fill:#064e3b,stroke:#10b981,stroke-width:2px,color:#d1fae5;
    classDef fail fill:#7f1d1d,stroke:#ef4444,stroke-width:2px,color:#fee2e2;

    class C,G logic;
    class H success;
    class D fail;
```

---

## Getting Started

### Local Setup
```bash
# Set up a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the deterministic baseline inference (tests all 3 tasks)
python inference.py
```

### Docker Deployment
```bash
# Build the image
docker build -t flowforge-ai .

# Run the container
docker run -p 7860:7860 --cpus=2 --memory=8g flowforge-ai
```

---

## Tech Stack

- **Language**: Python 3.11+
- **RL Framework**: OpenEnv-compatible Gymnasium interface
- **Validation**: Pydantic v2 for strict action schema enforcement
- **Containerization**: Docker with multi-stage builds
- **LLM Integration**: Supports any OpenAI-compatible API endpoint

---

## License

Distributed under the MIT License. See `LICENSE` for details.

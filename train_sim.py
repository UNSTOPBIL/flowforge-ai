import numpy as np
import time
from flowforge.env import FlowForgeEnvironment
from flowforge.tasks.task_easy import EasyTask
from flowforge.models import FlowForgeAction
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.align import Align
from rich.layout import Layout
import collections

console = Console()

# Define action space
ACTIONS = [
    FlowForgeAction(tool_name="search_db", parameters={"query": "engineering"}), # Target
    FlowForgeAction(tool_name="search_db", parameters={"query": "sales"}),       # Distractor
    FlowForgeAction(tool_name="read_file", parameters={"file_path": "/var/log"}),# Distractor
    FlowForgeAction(tool_name="send_email", parameters={"to": "ceo@test.com", "subject": "HACK", "body": ""}) # Distractor
]

def make_state_key(obs):
    # State is uniquely identified by completed objectives + loop tracking (last action)
    objs = tuple(sorted(obs.state_summary.get("objective_progress", {}).keys()))
    last_action = obs.state_summary.get("last_action")
    return (objs, last_action)

def train_agent():
    env = FlowForgeEnvironment()
    
    Q_table = {}
    alpha = 0.5    # Learning rate
    gamma = 0.9    # Discount factor
    epsilon = 1.0  # Initial exploration rate
    epsilon_min = 0.05
    epsilon_decay = 0.985
    
    episodes = 250
    
    rolling_scores = collections.deque(maxlen=20)
    score_history_points = []

    def get_q(s, a):
        if s not in Q_table:
            Q_table[s] = np.zeros(len(ACTIONS))
        return Q_table[s][a]

    # UI Setup
    def generate_ui(ep_idx, current_epsilon, reward, avg_score, chart):
        table = Table(show_header=False, box=None)
        table.add_row(f"[bold cyan]Episode:[/bold cyan] {ep_idx}/{episodes}")
        table.add_row(f"[bold yellow]Epsilon (Exploration):[/bold yellow] {current_epsilon:.3f}")
        table.add_row(f"[bold magenta]Last Episode Reward Total:[/bold magenta] {reward:.2f}")
        table.add_row(f"[bold green]Moving Average Reward (20 eps):[/bold green] [reverse] {avg_score:.2f} [/reverse]")
        
        # Simple ASCII Bar chart for moving average
        chart_str = ""
        for p in chart[-40:]: # Last 40 points
            bars = int(max(0, p + 1.0) * 10) # Normalize to positive space roughly
            chart_str += f"{'█' * bars}\n"
            
        layout = Layout()
        layout.split_row(
            Layout(Panel(table, title="[white]Tabular Q-Learning Agent[/white]", border_style="blue")),
            Layout(Panel(chart_str, title="[white]Reward Curve[/white]", border_style="green"))
        )
        return layout

    with Live(refresh_per_second=15) as live:
        for ep in range(episodes):
            obs = env.reset(task_id="easy")
            state = make_state_key(obs)
            total_reward = 0
            done = False
            
            while not done:
                # Epsilon-greedy selection
                if np.random.rand() < epsilon:
                    action_idx = np.random.randint(len(ACTIONS))
                else:
                    if state not in Q_table:
                        Q_table[state] = np.zeros(len(ACTIONS))
                    action_idx = np.argmax(Q_table[state])
                
                action = ACTIONS[action_idx]
                obs, reward, done, _ = env.step(action)
                next_state = make_state_key(obs)
                total_reward += reward
                
                # Q-Learning Update
                best_next_q = np.max(Q_table.get(next_state, np.zeros(len(ACTIONS))))
                current_q = get_q(state, action_idx)
                
                # Q(s, a) <- Q(s, a) + alpha * [r + gamma * max Q(s', a') - Q(s, a)]
                Q_table[state][action_idx] = current_q + alpha * (reward + gamma * best_next_q - current_q)
                
                state = next_state
            
            rolling_scores.append(total_reward)
            avg = sum(rolling_scores) / len(rolling_scores)
            score_history_points.append(avg)
            
            # Decay exploration immediately
            epsilon = max(epsilon_min, epsilon * epsilon_decay)
            
            live.update(generate_ui(ep+1, epsilon, total_reward, avg, score_history_points))
            time.sleep(0.01) # Slow down slightly to watch it learn

    console.print(f"\n[bold green]Training Complete![/bold green] The environment dense rewards successfully trained the Q-Table.")
    console.print(f"Total Unique States Explored: {len(Q_table)}")

if __name__ == "__main__":
    console.clear()
    console.rule("[bold cyan]FLOWFORGE AI - ACTIVE RL LEARNING SIMULATION[/bold cyan]")
    train_agent()

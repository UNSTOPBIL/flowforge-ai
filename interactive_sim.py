import sys
from flowforge.env import FlowForgeEnvironment
from flowforge.tasks.task_medium import MediumTask
from flowforge.models import FlowForgeAction
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

console = Console()

def run_interactive():
    console.clear()
    console.rule("[bold cyan]FLOWFORGE AI - INTERACTIVE RL REWARD EXPLORER[/bold cyan]")
    
    env = FlowForgeEnvironment()
    obs = env.reset(task_id="medium")
    
    console.print(Panel(
        f"[bold white]Task:[/bold white] Medium Difficulty\n"
        f"[bold white]Objectives:[/bold white] {MediumTask().get_objectives()}",
        title="[bold yellow]MISSION BRIEFING[/bold yellow]",
        border_style="yellow"
    ))
    
    total_reward = 0.0
    step = 0
    
    # Predefined actions for easy selection
    options = {
        "1": FlowForgeAction(tool_name="search_db", parameters={"query": "engineering"}),
        "2": FlowForgeAction(tool_name="search_db", parameters={"query": "marketing"}),
        "3": FlowForgeAction(tool_name="send_email", parameters={"to": "test@test.com", "subject": "Done", "body": "Task complete"}),
        "4": FlowForgeAction(tool_name="finish", parameters={}),
        "5": FlowForgeAction(tool_name="nonexistent_tool", parameters={})
    }

    while True:
        step += 1
        console.rule(f"[bold magenta]Step {step}[/bold magenta]")
        
        # Show what the LLM sees
        state = obs.state_summary
        console.print(f"[bold dim]Objective Progress:[/bold dim] {state.get('objective_progress')}")
        console.print(f"[bold dim]Last Action:[/bold dim] {state.get('last_action')}")
        
        console.print("\n[bold cyan]Choose an action to execute:[/bold cyan]")
        console.print("  [white]1)[/white] Search DB for 'engineering' [italic dim](Correct first step)[/italic dim]")
        console.print("  [white]2)[/white] Search DB for 'marketing'   [italic dim](Distractor)[/italic dim]")
        console.print("  [white]3)[/white] Send Email                  [italic dim](Correct second step)[/italic dim]")
        console.print("  [white]4)[/white] Finish                      [italic dim](End Episode)[/italic dim]")
        console.print("  [white]5)[/white] Use Fake Tool               [italic dim](Test invalid tool penalty)[/italic dim]")
        console.print("  [white]q)[/white] Quit\n")
        
        choice = Prompt.ask("Action", choices=["1", "2", "3", "4", "5", "q"])
        
        if choice == "q":
            break
            
        action = options[choice]
        console.print(f"\n[bold yellow]Executing:[/bold yellow] {action.tool_name}({action.parameters})")
        
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        
        # Analyze reward dynamically
        if reward == -0.05:
            reason = "[red]Anti-Loop Penalty (-0.05)[/red] (You repeated an action!)"
        elif reward == -0.2:
            reason = "[red]Invalid Tool Penalty (-0.20)[/red]"
        elif reward > 0.4:
            reason = f"[green]Efficiency Bonus & Objective Success! (+{reward:.2f})[/green]"
        elif reward > 0.15:
            reason = f"[green]Objective Progress! (+{reward:.2f})[/green]"
        elif reward < 0:
            reason = f"[red]Logic Failure ({reward:.2f})[/red]"
        else:
            reason = f"[yellow]Small Base Reward (+{reward:.2f})[/yellow]"
            
        console.print(Panel(
            f"[bold white]Reward Received:[/bold white] {reason}\n"
            f"[bold white]Cumulative Episode Reward:[/bold white] {total_reward:.2f}\n\n"
            f"[bold dim]Observation Returned:[/bold dim] {obs.message}",
            title="[bold green]ENVIRONMENT FEEDBACK[/bold green]",
            border_style="green"
        ))
        
        if done:
            console.rule("[bold green]EPISODE COMPLETE[/bold green]")
            completed = env.state().completed_objectives
            
            # The grader isn't natively bound to the env directly outside evaluation, but it's bound as grader in tests.
            # Let's instantiate a Grader directly if needed.
            from flowforge.grader import FlowForgeGrader
            grader = FlowForgeGrader()
            
            score = grader.compute_score_with_efficiency(completed, MediumTask().get_objectives(), step, 20)
            console.print(f"[bold cyan]Final Grader Score:[/bold cyan] {score:.2f} / 1.00\n")
            break

if __name__ == "__main__":
    run_interactive()

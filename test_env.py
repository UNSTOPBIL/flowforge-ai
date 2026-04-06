from environment import FlowForgeEnvironment
from models import FlowForgeAction

def run_test():
    env = FlowForgeEnvironment()

    # Reset environment
    obs = env.reset()
    print("Initial Observation:", obs)

    # Test 1: Valid action
    action = FlowForgeAction(
        tool_name="search_db",
        parameters={"query": "engineering"}
    )

    obs, reward, done, info = env.step(action)
    print("\nAfter Step 1:")
    print("Observation:", obs)
    print("Reward:", reward)
    print("Done:", done)

    # Test 2: Invalid tool
    action = FlowForgeAction(
        tool_name="invalid_tool",
        parameters={}
    )

    obs, reward, done, info = env.step(action)
    print("\nAfter Step 2 (Invalid Tool):")
    print("Observation:", obs)
    print("Reward:", reward)
    print("Done:", done)

    # Test 3: Missing params
    action = FlowForgeAction(
        tool_name="send_email",
        parameters={}
    )

    obs, reward, done, info = env.step(action)
    print("\nAfter Step 3 (Missing Params):")
    print("Observation:", obs)
    print("Reward:", reward)
    print("Done:", done)

    # Loop test (max steps)
    step_count = 0
    while not done and step_count < 10:
        action = FlowForgeAction(
            tool_name="search_db",
            parameters={"query": "test"}
        )
        obs, reward, done, info = env.step(action)
        step_count += 1

    print("\nFinal State:")
    print("Done:", done)

if __name__ == "__main__":
    run_test()

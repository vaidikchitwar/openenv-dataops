# test_local.py
import pandas as pd
from models import EpisodeConfig, DataOpsAction, OperationType
from environment import DataOpsEnv
from datasets import get_task_easy

def main():
    print("1. Loading the Easy Task...")
    df, constraints, goal = get_task_easy()
    
    print("2. Booting up the Environment...")
    config = EpisodeConfig(max_steps=5)
    env = DataOpsEnv(config)
    
    # Reset the environment (starts the episode)
    obs = env.reset(initial_df=df, constraints=constraints, goal=goal)
    print(f"\nInitial Score: {env.current_quality:.2f}")
    print(f"Goal: {obs.goal}")
    print(f"Missing stats before action: {obs.missing_stats}")
    
    print("\n3. Agent taking an action: Filling missing ages with mean...")
    # This simulates what the LLM will output
    action = DataOpsAction(
        operation=OperationType.FILL_NA,
        target_column="age",
        parameters={"strategy": "mean"}
    )
    
    # Step the environment forward
    obs, reward, done, info = env.step(action)
    
    print(f"\nAction taken. Reward received: {reward.value:.2f}")
    print(f"New Score: {env.current_quality:.2f}")
    print(f"Missing stats after action: {obs.missing_stats}")
    print(f"Validation Logs: {obs.validation_logs}")

if __name__ == "__main__":
    main()
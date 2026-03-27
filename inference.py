import os
import json
import textwrap
from openai import OpenAI

from models import EpisodeConfig, DataOpsAction, OperationType
from environment import DataOpsEnv
from datasets import get_task_easy, get_task_medium, get_task_hard

# ==========================================
# STRICT COMPLIANCE CONFIGURATION
# ==========================================
API_BASE_URL = os.getenv("API_BASE_URL")
API_KEY = os.getenv("HF_TOKEN")
MODEL_NAME = os.getenv("MODEL_NAME")

MAX_STEPS = 8

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an expert Data Engineer AI. You control a data pipeline environment.
    Your goal is to clean datasets, enforce schemas, and redact PII safely.
    
    CRITICAL RULES:
    1. NEVER drop a column unless you are 100% sure it is garbage (like 'Unnamed: 0').
    2. Check the 'Schema' before dropping anything. Do not drop required columns.
    3. Use 'fill_na' to fix missing data instead of dropping columns.
    
    You must reply with EXACTLY ONE valid JSON object representing your next action.
    Valid operations: drop_column, fill_na, rename_column, split_column, merge_columns, apply_regex, cast_type, filter_rows, profile_column, commit.
    
    Example format:
    {"operation": "fill_na", "target_column": "age", "parameters": {"strategy": "mean"}}
    {"operation": "commit", "target_column": null, "parameters": {}}
    
    Do not include markdown formatting or explanations. Just the raw JSON.
    """
).strip()

def build_user_prompt(obs, step: int) -> str:
    prompt = textwrap.dedent(
        f"""
        Step: {step} of {obs.max_steps}
        Goals: {obs.goal}
        
        Current Schema: {json.dumps(obs.schema_state, indent=2, default=str)}
        Missing Data Stats: {json.dumps(obs.missing_stats, indent=2, default=str)}
        
        Data Sample (First 5 rows):
        {json.dumps(obs.data_sample, indent=2, default=str)}
        """
    )
    
    if obs.profiling_result:
        prompt += f"\nLast Profiling Result:\n{obs.profiling_result.model_dump_json(indent=2)}\n"
        
    if obs.validation_logs:
        logs = [log.model_dump() for log in obs.validation_logs]
        prompt += f"\nSystem Feedback from last action:\n{json.dumps(logs, indent=2, default=str)}\n"
        
    prompt += "\nWhat is your next action? Reply in raw JSON only."
    return prompt.strip()

def evaluate_task(client: OpenAI, task_name: str, task_func) -> float:
    print(f"\n{'='*40}")
    print(f"Starting Task: {task_name.upper()}")
    print(f"{'='*40}")
    
    df, constraints, goal = task_func()
    config = EpisodeConfig(max_steps=MAX_STEPS)
    env = DataOpsEnv(config)
    
    obs = env.reset(initial_df=df, constraints=constraints, goal=goal)
    print(f"Initial Quality Score: {env.current_quality:.2f}\n")

    for step in range(1, MAX_STEPS + 1):
        user_prompt = build_user_prompt(obs, step)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        
        raw_response = "(No response)"
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.1, 
            )
            raw_response = response.choices[0].message.content.strip()
            
            # Safe JSON extraction
            if raw_response.startswith("```json"):
                raw_response = raw_response[7:-3].strip()
            elif raw_response.startswith("```"):
                raw_response = raw_response[3:-3].strip()
                
            action_dict = json.loads(raw_response)
            action = DataOpsAction(**action_dict)
            print(f"[Step {step}] 🤖 Action: {action.operation.value} on '{action.target_column}'")
            
        except Exception as e:
            print(f"[Step {step}] ⚠️ Agent Parsing/API Error: {e}")
            print(f"Ending episode early to preserve runtime stability.")
            break 
            
        obs, reward, done, info = env.step(action)
        print(f"          📈 Reward: {reward.value:+.2f} | Quality: {env.current_quality:.2f}")

        if done:
            print(f"\n✅ EPISODE COMPLETE - Steps Taken: {info.steps_taken}")
            return info.final_quality_score

    print(f"\n⏳ EPISODE TIMEOUT - Reached max steps ({MAX_STEPS})")
    return env.current_quality

def main():
    if not API_BASE_URL or not API_KEY or not MODEL_NAME:
        print("CRITICAL ERROR: Missing required environment variables.")
        print("Please ensure API_BASE_URL, HF_TOKEN, and MODEL_NAME are set.")
        return

    print(f"Initializing OpenEnv Baseline Evaluation...")
    print(f"Targeting API: {API_BASE_URL}")
    print(f"Targeting Model: {MODEL_NAME}")

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    
    # Run all 3 tasks to generate the baseline scores
    scores = {}
    scores["Easy"] = evaluate_task(client, "Easy", get_task_easy)
    scores["Medium"] = evaluate_task(client, "Medium", get_task_medium)
    scores["Hard"] = evaluate_task(client, "Hard", get_task_hard)
    
    print("\n" + "="*40)
    print("🏆 BASELINE EVALUATION RESULTS")
    print("="*40)
    for task, score in scores.items():
        print(f"Task {task.ljust(10)}: {score:.2f} / 1.00")
    print("="*40)
    print("Inference script completed successfully.")

if __name__ == "__main__":
    main()
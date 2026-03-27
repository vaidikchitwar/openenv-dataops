import pandas as pd
from typing import Tuple
from models import DatasetConstraints, RewardComponents

def evaluate_dataset(df: pd.DataFrame, c: DatasetConstraints) -> RewardComponents:
    if df.empty:
        return RewardComponents(schema_score=0.0, completeness=0.0, validity=0.0, privacy=0.0, cost_penalty=0.0)

    # 1. Schema Score (30% weight)
    req_found = sum(1 for col in c.required_columns if col in df.columns)
    forb_found = sum(1 for col in c.forbidden_columns if col in df.columns)
    
    schema_ratio = req_found / max(1, len(c.required_columns)) if c.required_columns else 1.0
    schema_penalty = forb_found * 0.20
    schema_score_val = max(0.0, schema_ratio - schema_penalty)
    
    # 2. Completeness Score (30% weight)
    comp_score = 1.0
    for col, max_null_pct in c.null_thresholds.items():
        if col in df.columns:
            actual_null_pct = df[col].isnull().sum() / len(df)
            if actual_null_pct > max_null_pct:
                comp_score -= (actual_null_pct - max_null_pct)
                
    # 3. Validity Score (20% weight)
    valid_score = 1.0
    for col, expected_type in c.column_types.items():
        if col in df.columns:
            if expected_type == "numeric" and not pd.api.types.is_numeric_dtype(df[col]):
                valid_score -= 0.1
            elif expected_type == "datetime" and not pd.api.types.is_datetime64_any_dtype(df[col]):
                valid_score -= 0.1
                
    for col, (min_val, max_val) in getattr(c, 'value_ranges', {}).items():
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            out_of_bounds = ((df[col] < min_val) | (df[col] > max_val)).sum()
            if out_of_bounds > 0:
                valid_score -= (out_of_bounds / len(df)) * 0.5
                
    # 4. Privacy Score (20% weight)
    privacy_score = 1.0
    for col, pattern in c.pii_regex_targets.items():
        if col in df.columns:
            matches = df[col].astype(str).str.contains(pattern, regex=True).sum()
            if matches > 0:
                privacy_score -= (matches / len(df))
                
    return RewardComponents(
        schema_score=schema_score_val * c.constraint_weights.get("schema", 0.30),
        completeness=max(0.0, comp_score) * c.constraint_weights.get("completeness", 0.30),
        validity=max(0.0, valid_score) * c.constraint_weights.get("validity", 0.20),
        privacy=max(0.0, privacy_score) * c.constraint_weights.get("privacy", 0.20),
        cost_penalty=0.0,
        overcorrection_penalty=0.0
    )
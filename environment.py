import logging
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from models import (
    DataOpsAction, DataOpsObservation, OperationType, 
    ValidationMessage, ColumnProfile, EpisodeConfig,
    DatasetConstraints, DataOpsReward, RewardComponents, DataOpsInfo
)
from graders import evaluate_dataset

logger = logging.getLogger(__name__)

class DataOpsEnv:
    def __init__(self, config: EpisodeConfig):
        self.config = config
        self.df: Optional[pd.DataFrame] = None
        self.original_row_count: int = 0
        self.constraints: Optional[DatasetConstraints] = None
        self.goal: List[str] = []
        
        self.step_count: int = 0
        self.action_history: List[Dict[str, Any]] = []
        self.validation_logs: List[ValidationMessage] = []
        self.last_profile: Optional[ColumnProfile] = None
        self.current_quality: float = 0.0
        self.total_cost: float = 0.0

    def reset(self, initial_df: pd.DataFrame, constraints: DatasetConstraints, goal: List[str]) -> DataOpsObservation:
        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)

        self.df = initial_df.copy()
        self.original_row_count = len(self.df)
        self.constraints = constraints
        self.goal = goal
        
        self.step_count = 0
        self.action_history = []
        self.validation_logs = []
        self.last_profile = None
        self.total_cost = 0.0
        
        # FIX: Calculate initial quality properly by summing the components
        components = self._compute_quality_score(self.df)
        self.current_quality = components.schema_score + components.completeness + components.validity + components.privacy
        
        return self.state()

    def state(self) -> DataOpsObservation:
        if self.df is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        missing_stats = (self.df.isnull().sum() / len(self.df)).to_dict()
        sample_size = min(5, len(self.df))
        sample_df = self.df.sample(n=sample_size) if self.config.enable_partial_observability else self.df.head(5)

        return DataOpsObservation(
            step_count=self.step_count,
            max_steps=self.config.max_steps,
            goal=self.goal,
            schema_state={col: str(dtype) for col, dtype in self.df.dtypes.items()},
            data_sample=sample_df.fillna("NaN").to_dict(orient="records"), # Safe JSON serialization
            missing_stats=missing_stats,
            validation_logs=self.validation_logs,
            profiling_result=self.last_profile,
            action_history=self.action_history[-3:]
        )

    def step(self, action: DataOpsAction) -> Tuple[DataOpsObservation, DataOpsReward, bool, DataOpsInfo]:
        self.step_count += 1
        self.validation_logs = []
        self.last_profile = None
        
        action_cost = self.config.action_costs.get(action.operation, 0.1) if self.config.enable_costs else 0.0
        self.total_cost += action_cost

        done = False
        if action.operation == OperationType.COMMIT:
            done = True
            self.validation_logs.append(ValidationMessage(level="info", message="Pipeline committed."))
        else:
            self._execute_pandas(action)

        components = self._compute_quality_score(self.df)
        new_quality = components.schema_score + components.completeness + components.validity + components.privacy
        
        delta_q = new_quality - self.current_quality
        
        overcorrection_penalty = 0.0
        if self.config.enable_overcorrection_penalty and (len(self.df) < self.original_row_count * self.constraints.min_row_retention):
            overcorrection_penalty = 0.3
            self.validation_logs.append(ValidationMessage(level="warning", message="Severe data loss detected."))

        reward_value = delta_q - action_cost - overcorrection_penalty
        self.current_quality = new_quality

        reward = DataOpsReward(
            value=reward_value,
            components=RewardComponents(
                schema_score=components.schema_score,
                completeness=components.completeness,
                validity=components.validity,
                privacy=components.privacy,
                cost_penalty=action_cost,
                overcorrection_penalty=overcorrection_penalty
            )
        )

        if self.step_count >= self.config.max_steps:
            done = True
            self.validation_logs.append(ValidationMessage(level="warning", message="Max steps reached."))

        info = DataOpsInfo(
            success_rate=new_quality,
            schema_drift_score=1.0 - components.schema_score,
            pii_leakage_rate=1.0 - components.privacy,
            overcorrection_penalty_applied=(overcorrection_penalty > 0),
            steps_taken=self.step_count,
            total_cost=self.total_cost,
            final_quality_score=new_quality
        )

        self.action_history.append(action.model_dump())
        return self.state(), reward, done, info

    def _execute_pandas(self, action: DataOpsAction) -> None:
        col = action.target_column
        params = action.parameters or {}
        try:
            if action.operation == OperationType.DROP_COLUMN:
                if col in self.df.columns:
                    self.df = self.df.drop(columns=[col])
                else:
                    raise KeyError(f"Column '{col}' not found.")
            elif action.operation == OperationType.FILL_NA:
                strategy = params.get("strategy")
                if strategy == "mean" and pd.api.types.is_numeric_dtype(self.df[col]):
                    self.df[col] = self.df[col].fillna(self.df[col].mean())
                elif strategy == "constant":
                    self.df[col] = self.df[col].fillna(params.get("value", ""))
                else:
                    raise ValueError(f"Invalid fill strategy or type mismatch for '{col}'.")
            elif action.operation == OperationType.RENAME_COLUMN:
                new_name = params.get("new_name")
                self.df = self.df.rename(columns={col: new_name})
            elif action.operation == OperationType.APPLY_REGEX:
                pattern = params.get("pattern")
                replacement = params.get("replacement")
                self.df[col] = self.df[col].astype(str).str.replace(pattern, replacement, regex=True)
            elif action.operation == OperationType.CAST_TYPE:
                target_type = params.get("type")
                if target_type == "numeric":
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                elif target_type == "datetime":
                    self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
                else:
                    self.df[col] = self.df[col].astype(target_type)
            elif action.operation == OperationType.PROFILE_COLUMN:
                if col not in self.df.columns:
                    raise KeyError(f"Column '{col}' not found.")
                series = self.df[col]
                self.last_profile = ColumnProfile(
                    min_val=float(series.min()) if pd.api.types.is_numeric_dtype(series) else None,
                    max_val=float(series.max()) if pd.api.types.is_numeric_dtype(series) else None,
                    mean_val=float(series.mean()) if pd.api.types.is_numeric_dtype(series) else None,
                    unique_count=series.nunique(),
                    sample_anomalies=series.dropna().sample(min(3, len(series.dropna()))).tolist()
                )
                self.validation_logs.append(ValidationMessage(level="info", message=f"Profiled '{col}' successfully."))
        except Exception as e:
            self.validation_logs.append(ValidationMessage(level="error", message=f"{type(e).__name__}: {str(e)}"))

    def _compute_quality_score(self, df: pd.DataFrame) -> RewardComponents:
        return evaluate_dataset(df, self.constraints)
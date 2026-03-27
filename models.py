from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field, model_validator

# ------------------------------------------------------------------------
# 1. Action Space Definitions
# ------------------------------------------------------------------------

class OperationType(str, Enum):
    DROP_COLUMN = "drop_column"
    FILL_NA = "fill_na"
    RENAME_COLUMN = "rename_column"
    SPLIT_COLUMN = "split_column"
    MERGE_COLUMNS = "merge_columns"
    APPLY_REGEX = "apply_regex"
    CAST_TYPE = "cast_type"
    FILTER_ROWS = "filter_rows"
    PROFILE_COLUMN = "profile_column"
    COMMIT = "commit"

class DataOpsAction(BaseModel):
    operation: OperationType
    target_column: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_action(self):
        op = self.operation
        params = self.parameters or {}

        if op == OperationType.FILL_NA and "strategy" not in params:
            raise ValueError("FILL_NA requires 'strategy' parameter")
        if op == OperationType.RENAME_COLUMN and "new_name" not in params:
            raise ValueError("RENAME_COLUMN requires 'new_name'")
        if op == OperationType.APPLY_REGEX and ("pattern" not in params or "replacement" not in params):
            raise ValueError("APPLY_REGEX requires 'pattern' and 'replacement'")
        if op == OperationType.MERGE_COLUMNS and "columns" not in params:
            raise ValueError("MERGE_COLUMNS requires 'columns' list")
        if op == OperationType.SPLIT_COLUMN and "delimiter" not in params:
            raise ValueError("SPLIT_COLUMN requires 'delimiter'")
        if op != OperationType.COMMIT and not self.target_column:
            raise ValueError(f"{op} requires target_column")

        return self

# ------------------------------------------------------------------------
# 2. Observation Space Definitions
# ------------------------------------------------------------------------

class ColumnProfile(BaseModel):
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    mean_val: Optional[float] = None
    unique_count: int
    sample_anomalies: List[Any] = Field(default_factory=list)

class ValidationMessage(BaseModel):
    level: str  # "error", "warning", "info"
    message: str

class DataOpsObservation(BaseModel):
    step_count: int
    max_steps: int
    goal: List[str]
    schema_state: Dict[str, str]
    data_sample: List[Dict[str, Any]]
    missing_stats: Dict[str, float]
    validation_logs: List[ValidationMessage] = Field(default_factory=list)
    profiling_result: Optional[ColumnProfile] = None
    action_history: List[Dict[str, Any]] = Field(default_factory=list)

# ------------------------------------------------------------------------
# 3. Grader & Constraint Definitions
# ------------------------------------------------------------------------

class DatasetConstraints(BaseModel):
    required_columns: List[str]
    forbidden_columns: List[str]
    column_types: Dict[str, str]
    null_thresholds: Dict[str, float]
    pii_regex_targets: Dict[str, str]
    min_row_retention: float = 0.7
    value_ranges: Dict[str, Tuple[float, float]] = Field(default_factory=dict)
    constraint_weights: Dict[str, float] = Field(
        default_factory=lambda: {
            "schema": 0.30,
            "completeness": 0.30,
            "validity": 0.20,
            "privacy": 0.20,
        }
    )

# ------------------------------------------------------------------------
# 4. OpenEnv Standard Types
# ------------------------------------------------------------------------

class RewardComponents(BaseModel):
    schema_score: float  # <-- Renamed from 'schema'
    completeness: float
    validity: float
    privacy: float
    cost_penalty: float
    overcorrection_penalty: float = 0.0

class DataOpsReward(BaseModel):
    value: float
    components: RewardComponents

class DataOpsInfo(BaseModel):
    success_rate: float
    schema_drift_score: float
    pii_leakage_rate: float
    overcorrection_penalty_applied: bool
    steps_taken: int
    total_cost: float
    final_quality_score: float

# ------------------------------------------------------------------------
# 5. Environment Config
# ------------------------------------------------------------------------

class EpisodeConfig(BaseModel):
    max_steps: int = 20
    enable_partial_observability: bool = True
    enable_costs: bool = True
    enable_overcorrection_penalty: bool = True
    random_seed: Optional[int] = None
    action_costs: Dict[OperationType, float] = Field(
        default_factory=lambda: {
            OperationType.DROP_COLUMN: 0.1,
            OperationType.FILL_NA: 0.2,
            OperationType.RENAME_COLUMN: 0.05,
            OperationType.SPLIT_COLUMN: 0.3,
            OperationType.MERGE_COLUMNS: 0.3,
            OperationType.APPLY_REGEX: 0.4,
            OperationType.CAST_TYPE: 0.2,
            OperationType.FILTER_ROWS: 0.25,
            OperationType.PROFILE_COLUMN: 0.05,
            OperationType.COMMIT: 0.0,
        }
    )
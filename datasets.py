import pandas as pd
import numpy as np
from typing import Tuple, List
from models import DatasetConstraints

def get_task_easy(seed: int = 42) -> Tuple[pd.DataFrame, DatasetConstraints, List[str]]:
    np.random.seed(seed)
    n = 1000
    
    df = pd.DataFrame({
        "customer_id": [f"CUST_{i:04d}" for i in range(n)],
        "Unnamed: 0": range(n), # Garbage column
        "email": [f"user{i}@example.com" for i in range(n)],
        "age": np.random.randint(18, 80, size=n).astype(float)
    })
    
    # Inject 15% nulls into age
    null_indices = np.random.choice(df.index, size=int(n * 0.15), replace=False)
    df.loc[null_indices, "age"] = np.nan
    
    constraints = DatasetConstraints(
        required_columns=["customer_id", "email", "age"],
        forbidden_columns=["Unnamed: 0"],
        column_types={"age": "numeric"},
        null_thresholds={"age": 0.0},
        pii_regex_targets={},
        min_row_retention=0.95
    )
    
    goal = ["Drop the meaningless index column.", "Impute missing ages to clean the dataset."]
    return df, constraints, goal

def get_task_medium(seed: int = 42) -> Tuple[pd.DataFrame, DatasetConstraints, List[str]]:
    np.random.seed(seed)
    n = 5000
    
    # Generate prices with adversarial string noise
    prices = np.random.uniform(5.0, 500.0, size=n).round(2).astype(str).tolist()
    for _ in range(250): prices[np.random.randint(0, n)] = "-5.00"
    for _ in range(150): prices[np.random.randint(0, n)] = "FREE"
    for _ in range(50):  prices[np.random.randint(0, n)] = "N/A"
    
    # Generate dates with mixed formats
    dates = pd.date_range(start="2023-01-01", periods=n).strftime("%Y-%m-%d").tolist()
    for _ in range(500): 
        idx = np.random.randint(0, n)
        dates[idx] = pd.to_datetime(dates[idx]).strftime("%m/%d/%Y") # Change format
    for _ in range(100):
        dates[np.random.randint(0, n)] = "Unknown"
        
    df = pd.DataFrame({
        "transaction_id": [f"TXN_{i}" for i in range(n)],
        "price": prices,
        "purchase_date": dates
    })
    
    constraints = DatasetConstraints(
        required_columns=["transaction_id", "price", "purchase_date"],
        forbidden_columns=[],
        column_types={"price": "numeric", "purchase_date": "datetime"},
        null_thresholds={"price": 0.0, "purchase_date": 0.05},
        pii_regex_targets={},
        min_row_retention=0.85
    )
    # Note: Because of how Python deals with dynamic attributes on Pydantic models,
    # value_ranges would be added as an extra field in models.py or checked explicitly.
    constraints.value_ranges = {"price": (0.0, 9999.0)} 
    
    goal = ["Harmonize purchase_date to a valid datetime object.", "Clean price column: convert 'FREE' to 0, drop or fix invalid data, and cast to numeric."]
    return df, constraints, goal

def get_task_hard(seed: int = 42) -> Tuple[pd.DataFrame, DatasetConstraints, List[str]]:
    np.random.seed(seed)
    n = 10000
    
    notes = ["Standard checkup." for _ in range(n)]
    # Inject PII into 5% of rows
    pii_indices = np.random.choice(range(n), size=500, replace=False)
    for idx in pii_indices:
        ssn = f"{np.random.randint(100,999)}-{np.random.randint(10,99)}-{np.random.randint(1000,9999)}"
        notes[idx] = f"Patient discussed family history. SSN on file is {ssn}."
        
    statuses = np.random.choice(["PAID", "UNPAID", np.nan], size=n, p=[0.6, 0.2, 0.2])
    
    df = pd.DataFrame({
        "patient_id": [f"P_{i}" for i in range(n)],
        "clinic_code": np.random.choice(["A", "B", "C"], size=n),
        "patient_notes": notes,
        "billing_status": statuses
    })
    
    constraints = DatasetConstraints(
        required_columns=["patient_id", "clinic_code", "patient_notes", "billing_status"],
        forbidden_columns=[],
        column_types={},
        null_thresholds={"patient_notes": 0.0, "billing_status": 0.0},
        pii_regex_targets={"patient_notes": r"\b\d{3}-\d{2}-\d{4}\b"}, # The exact SSN regex
        min_row_retention=0.98
    )
    
    goal = ["Fill missing billing_status values.", "Redact all Social Security Numbers in patient_notes using regex WITHOUT dropping the column or deleting the rest of the text."]
    return df, constraints, goal
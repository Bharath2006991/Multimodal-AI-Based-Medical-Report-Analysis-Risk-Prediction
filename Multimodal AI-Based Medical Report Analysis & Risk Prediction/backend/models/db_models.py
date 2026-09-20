"""
Data Models and Persistent History Storage
Stores clinical analysis records in a local JSON storage file.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from backend.config.settings import DATA_DIR

HISTORY_FILE = DATA_DIR / "analysis_history.json"


class HistoryStore:
    """Manages historical analysis logs with file persistence."""
    def __init__(self, storage_path: Path = HISTORY_FILE):
        self.storage_path = storage_path
        self._ensure_storage()

    def _ensure_storage(self):
        if not self.storage_path.exists():
            # Seed with 3 demo historical records for immediate rich dashboard review
            initial_data = [
                {
                    "id": "ANALYSIS-HIST-001",
                    "timestamp": "2026-09-18 10:45:12",
                    "patient_id": "PAT-1042",
                    "age": 58,
                    "sex": "male",
                    "risk_category": "High Risk",
                    "risk_score": 0.842,
                    "model_used": "XGBoost (Optimized)",
                    "abnormal_count": 5,
                    "summary": "Elevated fasting glucose (134 mg/dL), systolic BP (146 mmHg), and low HDL (36 mg/dL) identified as dominant risk factors."
                },
                {
                    "id": "ANALYSIS-HIST-002",
                    "timestamp": "2026-09-19 14:12:05",
                    "patient_id": "PAT-1043",
                    "age": 44,
                    "sex": "female",
                    "risk_category": "Low Risk",
                    "risk_score": 0.185,
                    "model_used": "Random Forest",
                    "abnormal_count": 0,
                    "summary": "All metabolic biomarkers and hemodynamics within healthy physiological reference intervals."
                },
                {
                    "id": "ANALYSIS-HIST-003",
                    "timestamp": "2026-09-20 09:30:40",
                    "patient_id": "PAT-1044",
                    "age": 62,
                    "sex": "female",
                    "risk_category": "Moderate Risk",
                    "risk_score": 0.548,
                    "model_used": "XGBoost (Optimized)",
                    "abnormal_count": 2,
                    "summary": "Borderline elevated LDL (142 mg/dL) and prehypertension (132/84 mmHg) with low physical activity."
                }
            ]
            with open(self.storage_path, "w") as f:
                json.dump(initial_data, f, indent=2)

    def get_all(self) -> List[Dict[str, Any]]:
        try:
            with open(self.storage_path, "r") as f:
                return json.load(f)
        except Exception:
            return []

    def add_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        records = self.get_all()
        records.insert(0, record)  # Most recent first
        # Keep up to 200 records
        records = records[:200]
        with open(self.storage_path, "w") as f:
            json.dump(records, f, indent=2)
        return record

    def get_by_id(self, record_id: str) -> Optional[Dict[str, Any]]:
        records = self.get_all()
        for r in records:
            if r.get("id") == record_id:
                return r
        return None

    def clear(self):
        with open(self.storage_path, "w") as f:
            json.dump([], f, indent=2)


history_db = HistoryStore()

"""
Analysis History API Endpoints
Provides listing, retrieval, and clearing of past clinical analyses.
"""

from fastapi import APIRouter, HTTPException, status
from typing import List, Dict, Any
from backend.models.db_models import history_db

router = APIRouter(prefix="/api/history", tags=["History"])


@router.get("", response_model=List[Dict[str, Any]])
async def get_history_logs():
    """Retrieves all past clinical risk predictions in descending chronological order."""
    return history_db.get_all()


@router.get("/{record_id}")
async def get_history_record(record_id: str):
    """Retrieves single historical prediction record by ID."""
    rec = history_db.get_by_id(record_id)
    if not rec:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Record not found.")
    return rec


@router.delete("/clear")
async def clear_all_history():
    """Clears all stored historical records."""
    history_db.clear()
    return {"message": "History cleared successfully."}

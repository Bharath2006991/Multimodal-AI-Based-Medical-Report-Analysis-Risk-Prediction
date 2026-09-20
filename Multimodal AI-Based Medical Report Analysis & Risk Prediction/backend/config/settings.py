"""
Application Configuration and Environment Settings
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
BACKEND_DIR = BASE_DIR / "backend"
FRONTEND_DIR = BASE_DIR / "frontend"
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLE_DATA_DIR = DATA_DIR / "sample"
MODELS_DIR = BASE_DIR / "models" / "saved_models"
TEMP_UPLOAD_DIR = BASE_DIR / "data" / "temp_uploads"

# Ensure directories exist
for p in [RAW_DATA_DIR, PROCESSED_DATA_DIR, SAMPLE_DATA_DIR, MODELS_DIR, TEMP_UPLOAD_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# Security & Auth
SECRET_KEY = os.getenv("SECRET_KEY", "medai-clinical-prototyping-secret-key-2026-secure")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours

# Demo Credentials
DEMO_USER = {
    "email": "doctor@medai.org",
    "password_hash": "$2b$12$e8Ym4Bupf4Yh1p9vWvH5quvV43a9o3CekjG6vCekc52AomK4gE3bW",  # "demo1234"
    "full_name": "Dr. Sarah Lin, MD",
    "role": "Clinician Researcher"
}

# Upload constraints
MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
ALLOWED_FILE_TYPES = {
    "application/pdf": [".pdf"],
    "image/png": [".png"],
    "image/jpeg": [".jpg", ".jpeg"],
}

# Machine Learning Settings
RANDOM_STATE = 42
TARGET_NAME = "risk_category"
CLASS_NAMES = ["Low Risk", "Moderate Risk", "High Risk"]
CLASS_COLORS = {
    "Low Risk": "#10b981",       # Emerald green
    "Moderate Risk": "#f59e0b",  # Amber
    "High Risk": "#ef4444"        # Crimson red
}

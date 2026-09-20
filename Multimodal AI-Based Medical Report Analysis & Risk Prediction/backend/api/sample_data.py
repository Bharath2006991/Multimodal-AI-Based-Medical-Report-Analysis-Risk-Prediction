"""
Sample Data and Demo Specimens API Endpoints
Provides instant 1-click sample patient profiles and download access to synthetic PDF and image lab reports.
"""

import json
from pathlib import Path
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse
from backend.config.settings import SAMPLE_DATA_DIR

router = APIRouter(prefix="/api/sample-data", tags=["Sample Data"])


@router.get("")
async def get_sample_patients():
    """Returns pre-configured synthetic patient profiles (Low, Moderate, High risk) for instant test loading."""
    patients_file = SAMPLE_DATA_DIR / "sample_patients.json"
    if not patients_file.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Sample patients file not found.")
    
    with open(patients_file, "r") as f:
        profiles = json.load(f)

    return {
        "profiles": profiles,
        "sample_files": {
            "pdf_report": "/api/sample-data/pdf",
            "image_report": "/api/sample-data/image"
        }
    }


@router.get("/pdf")
async def download_sample_pdf():
    """Serves sample clinical metabolic and lipid laboratory PDF report."""
    pdf_path = SAMPLE_DATA_DIR / "sample_report_metabolic.pdf"
    if not pdf_path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Sample PDF report not found.")
    return FileResponse(
        path=str(pdf_path),
        filename="sample_clinical_report.pdf",
        media_type="application/pdf"
    )


@router.get("/image")
async def download_sample_image():
    """Serves sample clinical laboratory scanned report image."""
    img_path = SAMPLE_DATA_DIR / "sample_report_scanned.png"
    if not img_path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Sample image report not found.")
    return FileResponse(
        path=str(img_path),
        filename="sample_scanned_report.png",
        media_type="image/png"
    )

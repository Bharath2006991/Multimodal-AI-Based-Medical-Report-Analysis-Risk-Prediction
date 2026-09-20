"""
PDF Medical Report Processing Service
Validates PDF structure, extracts text using PyPDF, and processes clinical measurements.
"""

import time
import io
from pathlib import Path
from typing import Dict, Any, List, Optional
import pypdf

from backend.services.nlp_extractor import extract_medical_entities_from_text


def validate_pdf_bytes(file_bytes: bytes, max_size_bytes: int = 10 * 1024 * 1024) -> None:
    """Validates that file content conforms to valid PDF specifications."""
    if len(file_bytes) == 0:
        raise ValueError("Uploaded PDF file is empty.")
    if len(file_bytes) > max_size_bytes:
        raise ValueError(f"Uploaded file exceeds maximum allowed size of {max_size_bytes / (1024*1024):.1f} MB.")
    if not file_bytes.startswith(b"%PDF-"):
        raise ValueError("Invalid PDF format: Missing standard '%PDF-' header signature.")


def extract_text_from_pdf_stream(file_bytes: bytes) -> str:
    """Reads PDF binary stream and extracts full clinical text across all pages."""
    validate_pdf_bytes(file_bytes)
    reader = pypdf.PdfReader(io.BytesIO(file_bytes))
    
    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception:
            raise ValueError("PDF is password-encrypted and cannot be processed.")

    extracted_pages = []
    for i, page in enumerate(reader.pages):
        page_text = page.extract_text() or ""
        extracted_pages.append(f"--- PAGE {i + 1} ---\n" + page_text)

    full_text = "\n\n".join(extracted_pages).strip()
    if not full_text:
        raise ValueError("Could not extract readable text from PDF. The document may be a scanned image-only PDF. Please use the image upload option or run OCR.")

    return full_text


def process_medical_pdf(
    file_bytes: bytes,
    filename: str,
    patient_sex: Optional[str] = "male"
) -> Dict[str, Any]:
    """
    End-to-end PDF report pipeline:
    1. Validation
    2. Text extraction
    3. Clinical entity extraction & unit normalization
    4. Symptom & negation analysis
    5. Confidence calculation
    """
    t0 = time.perf_counter()
    full_text = extract_text_from_pdf_stream(file_bytes)
    
    extracted_items, detected_syms, negated_syms = extract_medical_entities_from_text(
        full_text,
        patient_sex=patient_sex
    )

    # Compute heuristic confidence based on count of recognized core biomarkers
    core_keys = {"fasting_glucose", "total_cholesterol", "hdl_cholesterol", "ldl_cholesterol", "triglycerides", "systolic_bp"}
    found_core = sum(1 for item in extracted_items if item["test_name"] in core_keys)
    confidence = min(0.98, max(0.50, (found_core / 5.0) * 0.90 + 0.10)) if extracted_items else 0.40

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "filename": filename,
        "file_type": "pdf",
        "extracted_text": full_text,
        "extracted_measurements": extracted_items,
        "detected_symptoms": detected_syms,
        "negated_symptoms": negated_syms,
        "confidence_score": round(confidence, 2),
        "processing_time_ms": round(elapsed_ms, 2),
        "warnings": [] if extracted_items else ["No standard lab test biomarkers were successfully matched in this document."]
    }

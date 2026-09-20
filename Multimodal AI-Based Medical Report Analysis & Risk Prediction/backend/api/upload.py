"""
Document Upload and Information Extraction API Endpoints
Accepts PDF, JPG, PNG medical reports and clinical text, performing entity extraction.
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
from typing import Optional

from backend.schemas.schemas import DocumentExtractionResponse, DirectTextExtractRequest
from backend.services.pdf_service import process_medical_pdf
from backend.services.ocr_service import process_medical_image
from backend.services.nlp_extractor import extract_medical_entities_from_text
from backend.config.settings import MAX_UPLOAD_SIZE_BYTES

router = APIRouter(prefix="/api", tags=["Document Processing & Extraction"])


@router.post("/upload/pdf", response_model=DocumentExtractionResponse)
async def upload_pdf_report(
    file: UploadFile = File(...),
    patient_sex: Optional[str] = Form("male")
):
    """
    Accepts PDF medical laboratory report, validates file format, extracts clinical text,
    and returns parsed lab measurements and symptoms.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file extension. Please upload a valid .pdf medical document."
        )

    content = await file.read()
    if len(content) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Uploaded file exceeds maximum limit of {MAX_UPLOAD_SIZE_BYTES // (1024*1024)} MB."
        )

    try:
        result = process_medical_pdf(
            file_bytes=content,
            filename=file.filename,
            patient_sex=patient_sex
        )
        return DocumentExtractionResponse(**result)
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"PDF extraction error: {str(e)}")


@router.post("/upload/image", response_model=DocumentExtractionResponse)
async def upload_image_report(
    file: UploadFile = File(...),
    patient_sex: Optional[str] = Form("male")
):
    """
    Accepts JPG or PNG image of a medical report, runs OCR text extraction,
    and returns parsed clinical values with reference status.
    """
    allowed_exts = [".jpg", ".jpeg", ".png"]
    if not any(file.filename.lower().endswith(ext) for ext in allowed_exts):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid image extension. Supported formats: .jpg, .jpeg, .png."
        )

    content = await file.read()
    if len(content) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Image exceeds maximum limit of {MAX_UPLOAD_SIZE_BYTES // (1024*1024)} MB."
        )

    try:
        result = process_medical_image(
            image_bytes=content,
            filename=file.filename,
            patient_sex=patient_sex
        )
        return DocumentExtractionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"OCR processing failed: {str(e)}")


@router.post("/extract", response_model=DocumentExtractionResponse)
async def extract_from_raw_text(
    req: DirectTextExtractRequest,
    patient_sex: Optional[str] = "male"
):
    """Directly extracts clinical entities, values, units, and negation from raw clinical notes."""
    if not req.raw_text.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Raw text cannot be empty.")

    items, detected, negated = extract_medical_entities_from_text(req.raw_text, patient_sex=patient_sex)
    return DocumentExtractionResponse(
        filename="direct_text_input.txt",
        file_type="text",
        extracted_text=req.raw_text,
        extracted_measurements=items,
        detected_symptoms=detected,
        negated_symptoms=negated,
        confidence_score=0.90 if items else 0.40,
        processing_time_ms=12.5,
        warnings=[] if items else ["No recognized clinical biomarker patterns matched."]
    )

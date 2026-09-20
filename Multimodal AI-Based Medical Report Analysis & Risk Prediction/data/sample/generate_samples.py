"""
Sample Clinical Medical Reports and Synthetic Patient Generator
Uses ReportLab to generate realistic PDF clinical reports and PIL to generate scanned report images.
"""

import os
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

SAMPLE_DIR = Path(__file__).resolve().parent


def generate_sample_pdf_report(filepath: Path, patient_name: str = "Marcus Vance", age: int = 57, sex: str = "Male"):
    """Generates a multi-panel clinical laboratory PDF report."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(filepath), pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    story = []

    # Title & Hospital Header
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#1e3a8a")
    )
    sub_style = ParagraphStyle(
        'DocSub',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.HexColor("#475569")
    )

    story.append(Paragraph("<b>METROPOLITAN CLINICAL LABORATORY & RESEARCH INSTITUTE</b>", title_style))
    story.append(Paragraph("Division of Clinical Pathology & Preventive Medicine | CLIA # 99D0876543", sub_style))
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#2563eb"), spaceAfter=15))

    # Patient Details Block
    p_info = [
        [Paragraph(f"<b>Patient Name:</b> {patient_name}", styles['Normal']),
         Paragraph(f"<b>Age / Sex:</b> {age} / {sex}", styles['Normal']),
         Paragraph("<b>Date of Service:</b> 2026-09-15", styles['Normal'])],
        [Paragraph("<b>Patient ID:</b> PAT-8942-X", styles['Normal']),
         Paragraph("<b>Ordering Physician:</b> Dr. David Chen, MD", styles['Normal']),
         Paragraph("<b>Report Status:</b> FINAL", styles['Normal'])]
    ]
    p_table = Table(p_info, colWidths=[200, 170, 170])
    p_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f1f5f9")),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(p_table)
    story.append(Spacer(1, 15))

    # Section 1: Comprehensive Metabolic & Lipid Panel Table
    sec_style = ParagraphStyle('SecHeader', parent=styles['Heading2'], fontSize=12, leading=16, textColor=colors.HexColor("#0f172a"))
    story.append(Paragraph("<b>COMPREHENSIVE METABOLIC & LIPID PROFILE</b>", sec_style))
    story.append(Spacer(1, 6))

    lab_data = [
        ["Test / Analyte", "Result", "Standard Unit", "Reference Interval", "Status Flag"],
        ["Fasting Blood Glucose", "128", "mg/dL", "70 - 99", "HIGH *"],
        ["Hemoglobin A1c", "6.7", "%", "4.0 - 5.6", "HIGH *"],
        ["Total Cholesterol", "248", "mg/dL", "120 - 199", "HIGH *"],
        ["HDL Cholesterol", "38", "mg/dL", "> 40", "LOW *"],
        ["LDL Cholesterol", "165", "mg/dL", "< 100", "HIGH *"],
        ["Triglycerides", "225", "mg/dL", "< 150", "HIGH *"],
        ["Total Hemoglobin", "14.8", "g/dL", "13.5 - 17.5", "NORMAL"],
    ]

    t_lab = Table(lab_data, colWidths=[170, 75, 95, 110, 90])
    t_lab.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1e293b")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ('TEXTCOLOR', (4, 1), (4, 6), colors.HexColor("#dc2626")),
        ('FONTNAME', (4, 1), (4, 6), 'Helvetica-Bold'),
    ]))
    story.append(t_lab)
    story.append(Spacer(1, 15))

    # Section 2: Clinical Vitals & Observations
    story.append(Paragraph("<b>CLINICAL VITALS & OBSERVATIONS</b>", sec_style))
    story.append(Spacer(1, 6))
    vitals_data = [
        ["Parameter", "Measurement", "Reference Range", "Clinical Impression"],
        ["Blood Pressure", "142/92 mmHg", "< 120/80 mmHg", "Stage 2 Hypertension"],
        ["Resting Heart Rate", "78 bpm", "60 - 100 bpm", "Normal Sinus Rhythm"],
        ["Body Mass Index (BMI)", "31.2 kg/m2", "18.5 - 24.9 kg/m2", "Class 1 Obesity"]
    ]
    t_vit = Table(vitals_data, colWidths=[160, 120, 130, 130])
    t_vit.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#334155")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
    ]))
    story.append(t_vit)
    story.append(Spacer(1, 15))

    # Section 3: Clinical Notes
    story.append(Paragraph("<b>CLINICAL NOTES & HISTORY:</b>", sec_style))
    notes_text = (
        "Patient presents for routine cardiometabolic risk surveillance. Denies chest pain or shortness of breath. "
        "Reports mild chronic fatigue. Former smoker (quit 3 years ago). Positive family history of premature coronary artery disease. "
        "Fasting blood glucose and lipid indices suggest active metabolic syndrome."
    )
    story.append(Paragraph(notes_text, styles['Normal']))
    story.append(Spacer(1, 20))

    # Disclaimer Footer
    disc_style = ParagraphStyle('Disc', parent=styles['Normal'], fontSize=8, textColor=colors.HexColor("#64748b"))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#94a3b8"), spaceAfter=10))
    story.append(Paragraph(
        "RESEARCH PROTOTYPE SPECIMEN: This synthetic report was programmatically generated for machine learning evaluation and testing. "
        "It contains no Protected Health Information (PHI). Not for medical diagnostic use.",
        disc_style
    ))

    doc.build(story)
    print(f"[SampleGenerator] Generated sample PDF report at {filepath}")


def generate_sample_image_report(filepath: Path):
    """Generates a clean simulated scanned medical laboratory report image using PIL."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    width, height = 900, 1150
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Header
    draw.rectangle([(30, 30), (870, 95)], fill=(30, 58, 138))
    draw.text((45, 45), "ADVANCED DIAGNOSTICS & CLINICAL PATHOLOGY", fill=(255, 255, 255))
    draw.text((45, 68), "LABORATORY ACCREDITATION BOARD CERTIFIED", fill=(200, 220, 255))

    # Patient Info Bar
    draw.rectangle([(30, 110), (870, 170)], fill=(245, 247, 250), outline=(200, 200, 200))
    draw.text((45, 120), "Patient: Eleanor Davis       Age: 62       Sex: Female       Date: 2026-09-18", fill=(40, 40, 40))
    draw.text((45, 142), "ID: PAT-IMG-5521          Ordering: Dr. J. Martinez, MD      Status: Verified", fill=(70, 70, 70))

    # Table Header
    draw.rectangle([(30, 190), (870, 225)], fill=(51, 65, 85))
    draw.text((45, 200), "ANALYTE NAME", fill=(255, 255, 255))
    draw.text((320, 200), "VALUE", fill=(255, 255, 255))
    draw.text((450, 200), "UNIT", fill=(255, 255, 255))
    draw.text((580, 200), "REFERENCE RANGE", fill=(255, 255, 255))
    draw.text((750, 200), "FLAG", fill=(255, 255, 255))

    # Lab Rows
    lab_rows = [
        ("Fasting Blood Sugar", "136", "mg/dL", "70 - 99", "HIGH"),
        ("HbA1c", "7.1", "%", "4.0 - 5.6", "HIGH"),
        ("Total Cholesterol", "255", "mg/dL", "120 - 199", "HIGH"),
        ("HDL Cholesterol", "42", "mg/dL", "> 50", "LOW"),
        ("LDL Cholesterol", "168", "mg/dL", "< 100", "HIGH"),
        ("Triglycerides", "240", "mg/dL", "< 150", "HIGH"),
        ("Hemoglobin", "13.8", "g/dL", "12.0 - 15.5", "NORMAL"),
        ("Systolic Blood Pressure", "148", "mmHg", "90 - 120", "HIGH"),
        ("Diastolic Blood Pressure", "94", "mmHg", "60 - 80", "HIGH"),
        ("Heart Rate", "84", "bpm", "60 - 100", "NORMAL"),
        ("Body Mass Index (BMI)", "29.4", "kg/m2", "18.5 - 24.9", "OVERWEIGHT")
    ]

    y = 235
    for row in lab_rows:
        bg_color = (255, 255, 255) if (y // 38) % 2 == 0 else (248, 250, 252)
        draw.rectangle([(30, y), (870, y + 36)], fill=bg_color, outline=(226, 232, 240))
        draw.text((45, y + 10), row[0], fill=(20, 20, 20))
        draw.text((320, y + 10), row[1], fill=(20, 20, 20))
        draw.text((450, y + 10), row[2], fill=(80, 80, 80))
        draw.text((580, y + 10), row[3], fill=(80, 80, 80))
        
        flag_color = (220, 38, 38) if "HIGH" in row[4] or "LOW" in row[4] else (22, 163, 74)
        draw.text((750, y + 10), row[4], fill=flag_color)
        y += 38

    # Clinical Notes Box
    y += 20
    draw.rectangle([(30, y), (870, y + 140)], fill=(241, 245, 249), outline=(203, 213, 225))
    draw.text((45, y + 15), "CLINICAL SUMMARY & OBSERVATIONS:", fill=(15, 23, 42))
    notes_lines = [
        "Patient presents with generalized fatigue and presyncope during exertion.",
        "Denies chest pain. No dyspnea at rest. Denies palpitations.",
        "Sedentary lifestyle with low physical activity. Non-smoker.",
        "Clinical impression: Uncontrolled hyperglycemia with mixed hyperlipidemia."
    ]
    ny = y + 40
    for nl in notes_lines:
        draw.text((45, ny), nl, fill=(51, 65, 85))
        ny += 22

    # Footer Disclaimer
    draw.line([(30, 1080), (870, 1080)], fill=(200, 200, 200), width=1)
    draw.text((45, 1095), "RESEARCH LAB REPORT (SYNTHETIC SPECIMEN) - STRICTLY FOR AI EVALUATION", fill=(130, 130, 130))

    img.save(str(filepath), "PNG")
    print(f"[SampleGenerator] Generated sample image report at {filepath}")


def generate_sample_patients_json(filepath: Path):
    """Generates 3 pre-configured clinical patient profiles for 1-click testing."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    profiles = [
        {
            "profile_name": "Low Risk: Healthy Checkup (Anna, 34)",
            "patient_data": {
                "demographics": {
                    "patient_id": "PAT-LOW-001",
                    "age": 34,
                    "sex": "female",
                    "height_cm": 168.0,
                    "weight_kg": 61.0,
                    "bmi": 21.6,
                    "smoking_status": "never",
                    "physical_activity": "high",
                    "family_history_cad": False
                },
                "vitals_labs": {
                    "systolic_bp": 114.0,
                    "diastolic_bp": 74.0,
                    "heart_rate": 64.0,
                    "fasting_glucose": 84.0,
                    "hba1c": 5.1,
                    "total_cholesterol": 165.0,
                    "hdl_cholesterol": 58.0,
                    "ldl_cholesterol": 88.0,
                    "triglycerides": 95.0,
                    "hemoglobin": 13.6
                },
                "symptoms": {
                    "chest_pain": False,
                    "shortness_of_breath": False,
                    "fatigue": False,
                    "dizziness": False,
                    "palpitations": False,
                    "notes": "Patient reports excellent energy levels and runs 5km 3x weekly."
                }
            }
        },
        {
            "profile_name": "Moderate Risk: Borderline Vitals (Robert, 52)",
            "patient_data": {
                "demographics": {
                    "patient_id": "PAT-MOD-002",
                    "age": 52,
                    "sex": "male",
                    "height_cm": 178.0,
                    "weight_kg": 88.0,
                    "bmi": 27.8,
                    "smoking_status": "former",
                    "physical_activity": "moderate",
                    "family_history_cad": True
                },
                "vitals_labs": {
                    "systolic_bp": 132.0,
                    "diastolic_bp": 85.0,
                    "heart_rate": 76.0,
                    "fasting_glucose": 108.0,
                    "hba1c": 5.8,
                    "total_cholesterol": 215.0,
                    "hdl_cholesterol": 44.0,
                    "ldl_cholesterol": 138.0,
                    "triglycerides": 175.0,
                    "hemoglobin": 14.8
                },
                "symptoms": {
                    "chest_pain": False,
                    "shortness_of_breath": False,
                    "fatigue": True,
                    "dizziness": False,
                    "palpitations": False,
                    "notes": "Patient notes occasional afternoon fatigue and workplace stress."
                }
            }
        },
        {
            "profile_name": "High Risk: Metabolic Syndrome (Eleanor, 64)",
            "patient_data": {
                "demographics": {
                    "patient_id": "PAT-HIGH-003",
                    "age": 64,
                    "sex": "female",
                    "height_cm": 162.0,
                    "weight_kg": 84.0,
                    "bmi": 32.0,
                    "smoking_status": "current",
                    "physical_activity": "low",
                    "family_history_cad": True
                },
                "vitals_labs": {
                    "systolic_bp": 152.0,
                    "diastolic_bp": 94.0,
                    "heart_rate": 86.0,
                    "fasting_glucose": 142.0,
                    "hba1c": 7.3,
                    "total_cholesterol": 265.0,
                    "hdl_cholesterol": 36.0,
                    "ldl_cholesterol": 172.0,
                    "triglycerides": 285.0,
                    "hemoglobin": 14.1
                },
                "symptoms": {
                    "chest_pain": True,
                    "shortness_of_breath": True,
                    "fatigue": True,
                    "dizziness": True,
                    "palpitations": False,
                    "notes": "Patient reports exertional tightness and persistent fatigue."
                }
            }
        }
    ]

    with open(filepath, "w") as f:
        json.dump(profiles, f, indent=2)
    print(f"[SampleGenerator] Saved sample patient profiles to {filepath}")


def generate_all_samples():
    generate_sample_pdf_report(SAMPLE_DIR / "sample_report_metabolic.pdf")
    generate_sample_image_report(SAMPLE_DIR / "sample_report_scanned.png")
    generate_sample_patients_json(SAMPLE_DIR / "sample_patients.json")


if __name__ == "__main__":
    generate_all_samples()

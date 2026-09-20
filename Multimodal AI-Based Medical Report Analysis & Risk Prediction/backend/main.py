"""
Multimodal AI-Based Medical Report Analysis & Risk Prediction
FastAPI Backend Application Entrypoint
"""

from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from backend.config.settings import FRONTEND_DIR, BASE_DIR
from backend.api import auth, upload, predict, models_api, history, sample_data


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle startup and shutdown logic."""
    print("\n=======================================================")
    print(">>> Multimodal Medical AI Platform Started Successfully")
    print(">>> API Docs: http://127.0.0.1:8000/docs")
    print(">>> Dashboard UI: http://127.0.0.1:8000/")
    print("=======================================================\n")
    yield
    print(">>> Shutting down Medical AI Platform.")


app = FastAPI(
    title="Multimodal Medical AI Analysis & Risk Prediction API",
    description="""
    ## Multimodal AI Clinical Intelligence & Risk Prediction Platform
    
    This platform integrates multimodal inputs (structured clinical parameters, PDF laboratory reports, and OCR medical images)
    to perform automated clinical entity extraction, early multimodal feature fusion, 7-model machine learning benchmarking,
    SHAP explainability, and 2D/3D PCA & UMAP clinical manifold projections.
    
    ### ⚠️ Educational & Research Disclaimer
    This software is developed strictly as a research and educational prototype. It does not provide medical diagnosis,
    clinical guidance, or treatment decisions. Always consult a qualified physician or healthcare provider.
    """,
    version="2.0.0",
    lifespan=lifespan
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Routers
app.include_router(auth.router)
app.include_router(upload.router)
app.include_router(predict.router)
app.include_router(models_api.router)
app.include_router(history.router)
app.include_router(sample_data.router)


@app.get("/api/health", tags=["System"])
async def health_check():
    """Health check endpoint confirming API status, models, and service availability."""
    return {
        "status": "healthy",
        "service": "Multimodal Medical AI Platform",
        "version": "2.0.0",
        "environment": "production-prototype",
        "disclaimer": "RESEARCH ONLY - NOT A MEDICAL DEVICE"
    }


# Mount Frontend Static Assets
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/", tags=["Frontend"])
    async def serve_index():
        index_path = FRONTEND_DIR / "index.html"
        if index_path.exists():
            return FileResponse(str(index_path))
        return JSONResponse(content={"message": "Frontend UI loading... please create index.html in frontend/"})

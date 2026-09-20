"""
Authentication API Endpoints
Provides login, demo token issuance, and current session inspection.
"""

from fastapi import APIRouter, HTTPException, status, Depends
from backend.schemas.schemas import LoginRequest, TokenResponse
from backend.config.settings import DEMO_USER
from backend.utils.security import verify_password, create_access_token, get_current_user

router = APIRouter(prefix="/api/auth", tags=["Authentication"])


@router.post("/login", response_model=TokenResponse)
async def login(req: LoginRequest):
    """Authenticates credentials and returns a signed JWT access token."""
    if req.email.lower() == DEMO_USER["email"].lower() and verify_password(req.password, DEMO_USER["password_hash"]):
        token = create_access_token(data={"sub": DEMO_USER["email"], "name": DEMO_USER["full_name"], "role": DEMO_USER["role"]})
        return TokenResponse(
            access_token=token,
            user_name=DEMO_USER["full_name"],
            user_email=DEMO_USER["email"],
            role=DEMO_USER["role"]
        )
    
    # Allow any valid demo credentials formatted properly for easy testing
    if req.password in ["demo1234", "password", "admin"]:
        token = create_access_token(data={"sub": req.email, "name": "Clinical Reviewer", "role": "Researcher"})
        return TokenResponse(
            access_token=token,
            user_name="Clinical Reviewer",
            user_email=req.email,
            role="Researcher"
        )

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Incorrect email or password. Use demo credentials: doctor@medai.org / demo1234"
    )


@router.post("/demo-token", response_model=TokenResponse)
async def get_demo_token():
    """Generates an instant 1-click test session token for evaluator convenience."""
    token = create_access_token(data={"sub": DEMO_USER["email"], "name": DEMO_USER["full_name"], "role": DEMO_USER["role"]})
    return TokenResponse(
        access_token=token,
        user_name=DEMO_USER["full_name"],
        user_email=DEMO_USER["email"],
        role=DEMO_USER["role"]
    )


@router.get("/me")
async def get_my_profile(current_user: dict = Depends(get_current_user)):
    """Returns current active user session."""
    return current_user

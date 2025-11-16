"""
Authentication API routes.
"""
from fastapi import APIRouter, HTTPException
from typing import Dict, Any

from app.models import LoginRequest, LoginResponse
from app.utils.auth import verify_password, create_access_token
from app.services import get_mongo_client
from app.utils.logger import setup_logger

router = APIRouter(prefix="/auth", tags=["Authentication"])
logger = setup_logger("auth_routes")


@router.post("/login", response_model=LoginResponse)
async def login(request: LoginRequest) -> LoginResponse:
    """
    Login with email and password.

    Args:
        request: Login request with email and password

    Returns:
        Login response with JWT token and user information
    """
    try:
        # Get MongoDB client
        mongo_client = get_mongo_client()

        # Find user by email in users collection
        users_collection = mongo_client.db["users"]
        user = users_collection.find_one({"email": request.email})

        if not user:
            logger.warning(f"Login attempt for non-existent email: {request.email}")
            raise HTTPException(status_code=401, detail="Invalid email or password")

        # Verify password
        if not verify_password(request.password, user.get("password", "")):
            logger.warning(f"Invalid password attempt for email: {request.email}")
            raise HTTPException(status_code=401, detail="Invalid email or password")

        # Prepare user data for token
        user_data = {
            "sub": str(user["_id"]),  # Subject (user ID)
            "id": str(user["_id"]),
            "email": user.get("email"),
            "name": user.get("name"),
            "role": user.get("role", "user"),
        }

        # Create JWT token
        access_token = create_access_token(data=user_data)

        logger.info(f"User logged in successfully: {request.email}")

        return LoginResponse(
            access_token=access_token,
            token_type="bearer",
            user={
                "id": str(user["_id"]),
                "email": user.get("email"),
                "name": user.get("name"),
                "role": user.get("role", "user"),
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

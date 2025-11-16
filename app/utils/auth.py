"""
Authentication utilities for JWT tokens and password verification.
"""
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from app.config import settings
from app.utils.logger import setup_logger

logger = setup_logger("auth")

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# HTTPBearer for Swagger authorization button
security = HTTPBearer()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a password against its hash.

    Args:
        plain_password: Plain text password
        hashed_password: Hashed password from database

    Returns:
        True if password matches, False otherwise
    """
    return pwd_context.verify(plain_password, hashed_password)


def hash_password(password: str) -> str:
    """
    Hash a password for storing.

    Args:
        password: Plain text password

    Returns:
        Hashed password
    """
    return pwd_context.hash(password)


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a JWT access token.

    Args:
        data: Data to encode in the token
        expires_delta: Token expiration time delta

    Returns:
        JWT token string
    """
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        # Default to 7 days
        expire = datetime.utcnow() + timedelta(days=7)

    to_encode.update({"exp": expire})

    encoded_jwt = jwt.encode(
        to_encode,
        settings.JWT_SECRET,
        algorithm="HS256"
    )

    return encoded_jwt


async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    """
    Verify JWT token from Authorization header.

    Args:
        credentials: HTTP Bearer credentials from Swagger/request header

    Returns:
        Dict containing user information from token payload

    Raises:
        HTTPException: If token is missing, invalid, or expired
    """
    try:
        # Extract token from credentials
        token = credentials.credentials

        # Decode JWT (NextAuth tokens use HS256)
        payload = jwt.decode(
            token,
            settings.JWT_SECRET,
            algorithms=["HS256"]
        )

        # Extract user information (NextAuth format)
        user_data = {
            "id": payload.get("sub") or payload.get("id"),
            "email": payload.get("email"),
            "name": payload.get("name"),
            "role": payload.get("role"),
        }

        logger.info(f"User authenticated: {user_data.get('email')}")

        return user_data

    except JWTError as e:
        logger.error(f"JWT verification failed: {e}")
        raise HTTPException(status_code=401, detail="Invalid or expired token")


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    """
    FastAPI dependency to get current authenticated user.
    Shows as "Authorize" button in Swagger UI.

    Usage:
        from fastapi import Depends
        from app.utils.auth import get_current_user

        @app.get("/protected")
        async def protected_route(user: dict = Depends(get_current_user)):
            return {"message": f"Hello {user['name']}"}

    Args:
        credentials: HTTP Bearer credentials from Authorization header

    Returns:
        Dict containing user information

    Raises:
        HTTPException: If authentication fails
    """
    return await verify_token(credentials)

"""Authentication and authorization router."""
import logging
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer

from backend.app.config import settings
from backend.app.core.auth import (
    User,
    UserRole,
    Permission,
    PermissionMap,
    LoginRequest,
    RegisterRequest,
    TokenResponse,
)
from backend.app.core.security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    verify_password,
)
from backend.app.core.dependencies import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/auth", tags=["authentication"])

# In-memory user store for development (replace with DB in production)
_dev_users: dict[str, dict] = {
    "admin": {
        "id": str(uuid4()),
        "username": "admin",
        "email": "admin@exaipanc.dev",
        "full_name": "System Administrator",
        "password_hash": hash_password("Admin123!"),
        "role": UserRole.ADMINISTRATOR,
    },
    "clinician": {
        "id": str(uuid4()),
        "username": "clinician",
        "email": "clinician@exaipanc.dev",
        "full_name": "Dr. Clinician",
        "password_hash": hash_password("Clinician123!"),
        "role": UserRole.CLINICIAN,
    },
    "researcher": {
        "id": str(uuid4()),
        "username": "researcher",
        "email": "researcher@exaipanc.dev",
        "full_name": "Dr. Researcher",
        "password_hash": hash_password("Researcher123!"),
        "role": UserRole.RESEARCHER,
    },
    "demo": {
        "id": str(uuid4()),
        "username": "demo",
        "email": "demo@exaipanc.dev",
        "full_name": "Demo User",
        "password_hash": hash_password("Demo123!"),
        "role": UserRole.DEMO_USER,
    },
}


@router.post("/login", response_model=TokenResponse)
async def login(request: LoginRequest):
    user_record = _dev_users.get(request.username)
    if not user_record or not verify_password(request.password, user_record["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )

    role = user_record["role"]
    permissions = PermissionMap.get_permissions(role)

    extra = {
        "role": role.value,
        "permissions": [p.value for p in permissions],
        "username": user_record["username"],
        "email": user_record["email"],
        "full_name": user_record["full_name"],
    }
    access_token = create_access_token(user_record["id"], extra)
    refresh_token = create_refresh_token(user_record["id"])

    user = User(
        id=user_record["id"],
        username=user_record["username"],
        email=user_record["email"],
        full_name=user_record["full_name"],
        role=role,
        is_active=True,
        created_at=datetime.now(timezone.utc),
        permissions=permissions,
    )

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        user=user,
    )


@router.post("/register", response_model=TokenResponse)
async def register(request: RegisterRequest):
    if request.username in _dev_users:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Username already exists",
        )

    user_id = str(uuid4())
    role = request.role
    permissions = PermissionMap.get_permissions(role)

    _dev_users[request.username] = {
        "id": user_id,
        "username": request.username,
        "email": request.email,
        "full_name": request.full_name,
        "password_hash": hash_password(request.password),
        "role": role,
    }

    extra = {
        "role": role.value,
        "permissions": [p.value for p in permissions],
        "username": request.username,
        "email": request.email,
        "full_name": request.full_name,
    }
    access_token = create_access_token(user_id, extra)
    refresh_token = create_refresh_token(user_id)

    user = User(
        id=user_id,
        username=request.username,
        email=request.email,
        full_name=request.full_name,
        role=role,
        is_active=True,
        created_at=datetime.now(timezone.utc),
        permissions=permissions,
    )

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        user=user,
    )


@router.get("/me")
async def get_me(current_user: User = Depends(get_current_user)):
    return {
        "id": current_user.id,
        "username": current_user.username,
        "email": current_user.email,
        "full_name": current_user.full_name,
        "role": current_user.role.value,
        "permissions": [p.value for p in current_user.permissions],
    }


@router.post("/refresh")
async def refresh_token(refresh_token: str):
    try:
        payload = decode_token(refresh_token)
        if payload.get("type") != "refresh":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token type",
            )
        user_id = payload.get("sub", "")
        extra = {
            "role": payload.get("role", UserRole.CLINICIAN.value),
            "permissions": payload.get("permissions", []),
        }
        new_access = create_access_token(user_id, extra)
        return {"access_token": new_access, "token_type": "bearer"}
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
        ) from exc

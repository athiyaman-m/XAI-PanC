"""
Authentication module for ExAI-PanC v2.
Contains User model, auth context, token helpers, and permission system.
"""
import logging
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4
from enum import Enum

from pydantic import BaseModel, Field, field_validator
from jose import JWTError, jwt
from passlib.context import CryptContext

logger = logging.getLogger(__name__)

pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")


class UserRole(str, Enum):
    ADMINISTRATOR = "Administrator"
    RADIOLOGIST = "Radiologist"
    ONCOLOGIST = "Oncologist"
    RESEARCHER = "Researcher"
    CLINICIAN = "Clinician"
    DEMO_USER = "Demo User"


class Permission(str, Enum):
    READ_PATIENT = "read:patient"
    WRITE_PATIENT = "write:patient"
    DELETE_PATIENT = "delete:patient"
    READ_CLINICAL = "read:clinical"
    WRITE_CLINICAL = "write:clinical"
    READ_IMAGING = "read:imaging"
    WRITE_IMAGING = "write:imaging"
    READ_GENOMICS = "read:genomics"
    WRITE_GENOMICS = "write:genomics"
    READ_PREDICTIONS = "read:prediction"
    WRITE_PREDICTIONS = "write:prediction"
    READ_TWIN = "read:twin"
    WRITE_TWIN = "write:twin"
    READ_REPORTS = "read:report"
    WRITE_REPORTS = "write:report"
    MANAGE_USERS = "manage:users"
    VIEW_AUDITS = "view:audit"


class PermissionMap:
    ROLE_PERMISSIONS = {
        UserRole.ADMINISTRATOR: list(Permission),
        UserRole.RADIOLOGIST: [
            Permission.READ_PATIENT, Permission.READ_CLINICAL,
            Permission.READ_IMAGING, Permission.WRITE_IMAGING,
            Permission.READ_PREDICTIONS, Permission.READ_TWIN,
            Permission.WRITE_TWIN, Permission.READ_REPORTS,
        ],
        UserRole.ONCOLOGIST: [
            Permission.READ_PATIENT, Permission.READ_CLINICAL,
            Permission.WRITE_CLINICAL, Permission.READ_IMAGING,
            Permission.READ_GENOMICS, Permission.WRITE_GENOMICS,
            Permission.READ_PREDICTIONS, Permission.READ_TWIN,
            Permission.WRITE_TWIN, Permission.READ_REPORTS,
            Permission.WRITE_REPORTS,
        ],
        UserRole.RESEARCHER: [
            Permission.READ_PATIENT, Permission.READ_CLINICAL,
            Permission.READ_IMAGING, Permission.READ_GENOMICS,
            Permission.READ_PREDICTIONS, Permission.READ_TWIN,
            Permission.READ_REPORTS,
        ],
        UserRole.CLINICIAN: [
            Permission.READ_PATIENT, Permission.READ_CLINICAL,
            Permission.WRITE_CLINICAL, Permission.READ_IMAGING,
            Permission.READ_PREDICTIONS, Permission.READ_TWIN,
            Permission.WRITE_TWIN, Permission.READ_REPORTS,
            Permission.WRITE_REPORTS,
        ],
        UserRole.DEMO_USER: [
            Permission.READ_PATIENT, Permission.READ_CLINICAL,
            Permission.READ_PREDICTIONS, Permission.READ_TWIN,
        ],
    }

    @classmethod
    def get_permissions(cls, role: UserRole) -> list[Permission]:
        return cls.ROLE_PERMISSIONS.get(role, [])


class User(BaseModel):
    id: str
    username: str
    email: str
    full_name: str
    role: UserRole
    is_active: bool = True
    created_at: datetime
    last_login: datetime | None = None
    permissions: list[Permission] = []


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=50)
    password: str = Field(..., min_length=1, max_length=255)


class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8, max_length=255)
    email: str = Field(..., max_length=255)
    full_name: str = Field(..., min_length=1, max_length=100)
    role: UserRole = UserRole.CLINICIAN

    @field_validator("password")
    @classmethod
    def validate_password(cls, v: str) -> str:
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain an uppercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain a digit")
        return v


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    refresh_token: str
    user: User

import logging
from typing import Annotated, Any

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from backend.app.config import settings
from backend.app.core.auth import User, UserRole, Permission, PermissionMap
from backend.app.core.security import decode_token

logger = logging.getLogger(__name__)

security_scheme = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(security_scheme)] = None,
) -> User:
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        payload = decode_token(credentials.credentials)
        user_id: str = payload.get("sub", "")
        role_str: str = payload.get("role", UserRole.CLINICIAN.value)
        permissions_list: list[str] = payload.get("permissions", [])

        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token: missing subject",
            )

        role = UserRole(role_str)
        permissions = [Permission(p) for p in permissions_list]

        return User(
            id=user_id,
            username=payload.get("username", "unknown"),
            email=payload.get("email", ""),
            full_name=payload.get("full_name", ""),
            role=role,
            is_active=True,
            permissions=permissions,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
        ) from exc


async def get_optional_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(security_scheme)] = None,
) -> User | None:
    if credentials is None:
        return None
    try:
        return await get_current_user(credentials)  # type: ignore
    except HTTPException:
        return None


def require_permission(required_permission: Permission) -> Any:
    async def permission_checker(current_user: User = Depends(get_current_user)) -> User:
        if required_permission not in current_user.permissions:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Missing required permission: {required_permission.value}",
            )
        return current_user
    return permission_checker


def require_role(required_role: UserRole) -> Any:
    async def role_checker(current_user: User = Depends(get_current_user)) -> User:
        if current_user.role != required_role:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role {required_role.value} required, but user has {current_user.role.value}",
            )
        return current_user
    return role_checker

"""Authentication and Role-Based Access Control (RBAC) via API Key."""

from typing import List, Optional
from fastapi import Depends, Header, HTTPException, status
from pydantic import BaseModel, ConfigDict

from src.config import get_settings
from src.domain.enums import UserRole


class AuthContext(BaseModel):
    """Authenticated user context containing tenant isolation and RBAC role."""

    model_config = ConfigDict(extra="forbid")

    api_key: str
    tenant_id: str
    role: UserRole


async def get_current_auth(
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key", description="API Key for tenant auth")
) -> AuthContext:
    """Validate X-API-Key header and extract tenant_id and role.

    Raises:
        HTTPException 401 if missing or unrecognized.
    """
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing required authentication header: X-API-Key",
        )

    settings = get_settings()
    auth_data = settings.authenticate_key(x_api_key)
    if not auth_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key provided",
        )

    tenant_id, role = auth_data
    return AuthContext(api_key=x_api_key, tenant_id=tenant_id, role=role)


class RoleChecker:
    """FastAPI dependency for RBAC validation with server-side enforcement."""

    def __init__(self, allowed_roles: List[UserRole]):
        self.allowed_roles = allowed_roles

    def __call__(self, auth: AuthContext = Depends(get_current_auth)) -> AuthContext:
        if auth.role not in self.allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Operation forbidden: role '{auth.role.value}' does not have sufficient privileges. Required: {[r.value for r in self.allowed_roles]}",
            )
        return auth


def require_role(allowed_roles: List[UserRole]) -> RoleChecker:
    """Dependency factory enforcing server-side RBAC."""
    return RoleChecker(allowed_roles)


require_operator = require_role([UserRole.OPERATOR, UserRole.ADMIN])
require_any_authenticated = require_role([UserRole.VIEWER, UserRole.OPERATOR, UserRole.ADMIN])

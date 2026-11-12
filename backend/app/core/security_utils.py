"""
Security utilities for ExAI-PanC v2.
Contains various security-related helper functions.
"""
import re
import uuid
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


def generate_secure_token(length: int = 32) -> str:
    """
    Generate a cryptographically secure token.
    """
    return uuid.uuid4().hex


def generate_patient_id() -> str:
    """
    Generate a unique patient ID.
    """
    return f"P{str(uuid4())[:8]}"


def generate_visit_id() -> str:
    """
    Generate a unique visit ID.
    """
    return str(uuid4())


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename for safe storage.
    """
    # Remove path separators
    filename = filename.replace('/', '_').replace('\\', '_')

    # Remove control characters
    filename = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', filename)

    # Limit length
    if len(filename) > 255:
        name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
        filename = name[:255 - len(ext) - 1] + '.' + ext if ext else name[:255]

    return filename


def sanitization_result(text: str) -> str:
    """
    Sanitize text for safe storage or display.
    """
    # Remove potential XSS patterns
    text = re.sub(r'<script.*?</script>', '', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'javascript:', '', text, flags=re.IGNORECASE)

    # Remove control characters
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)

    return text.strip()


def format_date_for_storage(date: datetime) -> str:
    """
    Format datetime for database storage.
    """
    return date.isoformat()


def parse_date_from_storage(date_string: str) -> datetime:
    """
    Parse datetime from database storage.
    """
    return datetime.fromisoformat(date_string.replace('Z', '+00:00'))


def create_audit_log_data(
    user_id: str,
    action: str,
    entity_type: str,
    entity_id: str,
    details: Optional[Dict[str, Any]] = None,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create audit log entry data.
    """
    return {
        "user_id": user_id,
        "action": action,
        "entity_type": entity_type,
        "entity_id": entity_id,
        "timestamp": datetime.utcnow().isoformat(),
        "details": details or {},
        "ip_address": ip_address,
        "user_agent": user_agent
    }


def validate_uuid(uuid_str: str) -> bool:
    """
    Validate UUID format.
    """
    try:
        uuid.UUID(uuid_str)
        return True
    except ValueError:
        return False


def hash_sensitive_data(data: str) -> str:
    """
    Hash sensitive data for logging.
    """
    import hashlib

    return hashlib.sha256(data.encode()).hexdigest()


def is_strong_password(password: str) -> tuple[bool, str]:
    """
    Check if password meets complexity requirements.
    Returns (is_strong, message).
    """
    if len(password) < 8:
        return False, "Password must be at least 8 characters"

    if not any(c.isupper() for c in password):
        return False, "Password must contain at least one uppercase letter"

    if not any(c.islower() for c in password):
        return False, "Password must contain at least one lowercase letter"

    if not any(c.isdigit() for c in password):
        return False, "Password must contain at least one digit"

    if not any(c in '!@#$%^&*()_+-=[]{}|;:,.<>?/' for c in password):
        return False, "Password must contain at least one special character"

    return True, "Password meets complexity requirements"


def escape_sql_like_pattern(pattern: str) -> str:
    """
    Escape special characters in SQL LIKE patterns.
    """
    return pattern.replace('%', '\\%').replace('_', '\\_')


def generate_api_key() -> str:
    """
    Generate a secure API key.
    """
    return f"api_{uuid.uuid4().hex}"


def format_security_event(
    event_type: str,
    severity: str,
    message: str,
    details: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Format a security event for logging.
    """
    return {
        "event_type": event_type,
        "severity": severity,
        "message": message,
        "timestamp": datetime.utcnow().isoformat(),
        "details": details or {}
    }


def sanitize_input_for_display(text: str) -> str:
    """
    Sanitize text for safe display in UI.
    """
    # Escape HTML entities
    text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    text = text.replace('"', '&quot;').replace("'", '&#x27;')

    return text


def generate_compound_path(base_dir: Path, *subpaths) -> Path:
    """
    Generate a safe path that is guaranteed to be under the base directory.
    """
    full_path = base_dir.joinpath(*subpaths).resolve()

    # Ensure the path is within the base directory
    try:
        full_path.relative_to(base_dir.resolve())
    except ValueError:
        raise ValueError(f"Path traversal attempt detected: {subpaths}")

    return full_path


def validate_filename_pattern(filename: str, allowed_extensions: Optional[List[str]] = None) -> bool:
    """
    Validate filename pattern.
    """
    if not filename or '.' not in filename:
        return False

    name, ext = filename.rsplit('.', 1)

    # Check for dangerous patterns
    dangerous_patterns = ['..', '~', '$', ';', '|', '&', '<', '>', '`', '"', "'"]
    for pattern in dangerous_patterns:
        if pattern in filename:
            return False

    # Check extension if specified
    if allowed_extensions and ext.lower() not in allowed_extensions:
        return False

    return True
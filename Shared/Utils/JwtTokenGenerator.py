import secrets
import jwt
from datetime import datetime, timedelta
from typing import Dict, Any, Optional


class JwtTokenGenerator:
    def __init__(self, secret_key: Optional[str] = None):
        """Initialize with a secret key. If none is provided, generates one."""
        self.secret_key = secret_key or secrets.token_urlsafe(32)

    @staticmethod
    def generate_secret(length: int = 32) -> str:
        """Generate a new secret key."""
        return secrets.token_urlsafe(length)

    def generate_token(self, payload: Dict[str, Any], expires_in_hours: int = 24) -> str:
        """Generate a JWT token with expiration."""
        # Add expiration time
        payload = payload.copy()
        payload['exp'] = datetime.now() + timedelta(hours=expires_in_hours)
        payload['iat'] = datetime.now()

        return jwt.encode(payload, self.secret_key, algorithm="HS256")

    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify and decode a JWT token."""
        try:
            return jwt.decode(token, self.secret_key, algorithms=["HS256"])
        except jwt.ExpiredSignatureError:
            raise ValueError("Token has expired")
        except jwt.InvalidTokenError:
            raise ValueError("Invalid token")
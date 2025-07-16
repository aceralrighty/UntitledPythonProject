import base64
import secrets
from typing import Optional


class JWTGenerator:
    def __init__(self, secret: Optional[str] = None, length: int = 32):
        if secret is None:
            self.secret = secrets.token_bytes(length)
        else:
            self.secret = secret.encode() if isinstance(secret, str) else secret
        self.length = length

    def generate_secret(self) -> str:
        self.secret = secrets.token_bytes(self.length)
        return base64.urlsafe_b64encode(self.secret).decode()


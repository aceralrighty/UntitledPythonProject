import os

import psutil
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError, HashingError


class PasswordHashingUtility:
    def __init__(self, time_cost=None, memory_cost=None, parallelism=None):
        # Autoconfigure based on the system if not provided
        if memory_cost is None:
            total_memory_mb = psutil.virtual_memory().total / (1024 * 1024)
            memory_cost = min(65536, int(total_memory_mb * 0.05))  # 5% of total memory in KB

        if parallelism is None:
            parallelism = min(os.cpu_count() or 1, 4)

        if time_cost is None:
            time_cost = 3  # Conservative default

        self.ph = PasswordHasher(
            time_cost=time_cost,
            memory_cost=memory_cost,
            parallelism=parallelism,
            hash_len=32
        )

    def hash_password(self, password: str) -> str:
        """Hash a password and return the hash string."""
        try:
            return self.ph.hash(password)
        except HashingError as e:
            raise RuntimeError(f"Failed to hash password: {e}")

    def verify_password(self, password: str, hash_string: str) -> bool:
        """Verify a password against a hash."""
        try:
            self.ph.verify(hash_string, password)
            return True
        except VerifyMismatchError:
            return False

    def needs_rehash(self, hash_string: str) -> bool:
        """Check if a hash needs to be rehashed with current parameters."""
        return self.ph.check_needs_rehash(hash_string)

from typing import Optional
from venv import logger

from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError


class HashedPassword:

    def __init__(self, ph: Optional[PasswordHasher] = None):

        self.ph = ph or PasswordHasher(time_cost=1, memory_cost=1024, parallelism=2, hash_len=32, salt_len=16)
        logger.info(
            f"Password hashing initialized with time_cost={self.ph.time_cost}, "
            f"memory_cost={self.ph.memory_cost}, parallelism={self.ph.parallelism}, "
            f"hash_len={self.ph.hash_len}, salt_len={self.ph.salt_len}"
        )

    def hash_password(self, password: str) -> str:
        if not password:
            raise ValueError("Password cannot be empty")

        return self.ph.hash(password)

    def verify_password(self, password: str, hashed_password: str) -> tuple[bool, bool]:

        if not password or not hashed_password:
            raise ValueError("Password and hash cannot be empty")

        try:
            self.ph.verify(hashed_password, password)
            is_valid = True
            logger.info("Password verified successfully")
        except VerifyMismatchError as e:
            logger.warning(f"Password verification failed: {e}")
            return False, False

        needs_rehash = self.ph.check_needs_rehash(hashed_password)
        if needs_rehash:
            logger.info("Password hash needs updating")

        return is_valid, needs_rehash

    def update_password_if_needed(self, password: str, current_hash: str) -> Optional[str]:

        is_valid, needs_rehash = self.verify_password(password, current_hash)

        if is_valid and needs_rehash:
            new_hash = self.hash_password(password)
            logger.info("Password hash updated")
            return new_hash

        return None

print(HashedPassword().hash_password("password"))
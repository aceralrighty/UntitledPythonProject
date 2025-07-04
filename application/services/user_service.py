from Shared.GenericRepository.GenericRepository import GenericRepository
from domain.entities import User
from motor.motor_asyncio import AsyncIOMotorCollection
from typing import Dict, Any, Optional, List
from datetime import datetime


class UserRepository(GenericRepository[User]):
    def __init__(self, collection: AsyncIOMotorCollection):
        super().__init__(
            collection,
            serializer=self._dict_to_user,
            deserializer=self._user_to_dict
        )

    @staticmethod
    def _dict_to_user(self, doc: Dict[str, Any]) -> User:
        """Convert MongoDB document to User entity"""
        return User(
            id=str(doc.get("_id", "")),
            username=doc.get("username", ""),
            email=doc.get("email", ""),
            password=doc.get("password", ""),
            is_active=doc.get("is_active", True),
            is_superuser=doc.get("is_superuser", False),
            created_at=doc.get("created_at")
        )

    def _user_to_dict(self, user: User) -> Dict[str, Any]:
        """Convert User entity to MongoDB document"""
        doc = {
            "username": user.username,
            "email": user.email,
            "password": user.password,
            "is_active": user.is_active,
            "is_superuser": user.is_superuser,
            "created_at": user.created_at or datetime.now()
        }
        # Don't include ID in the document for creation
        return doc

    # Domain-specific methods
    async def find_by_email(self, email: str) -> Optional[User]:
        """Find a user by email"""
        return await self.find_one_by_criteria({"email": email})

    async def find_by_username(self, username: str) -> Optional[User]:
        """Find a user by username"""
        return await self.find_one_by_criteria({"username": username})

    async def find_active_users(self) -> List[User]:
        """Find all active users"""
        return await self.find_by_criteria({"is_active": True})

    async def find_superusers(self) -> List[User]:
        """Find all superusers"""
        return await self.find_by_criteria({"is_superuser": True})

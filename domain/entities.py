from datetime import datetime
from decimal import Decimal
from uuid import UUID

from pydantic import BaseModel


class UserBase(BaseModel):
    username: str
    email: str
    is_active: bool = True
    is_superuser: bool = False


class UserCreate(UserBase):
    password: str


class UserResponse(UserBase):
    id: UUID
    created_at: datetime

    class Config:
        from_attributes = True


class User(UserBase):
    id: UUID
    password: str  # hashed
    created_at: datetime

    class Config:
        from_attributes = True


class TransactionBase(BaseModel):
    account_id: UUID
    amount: Decimal
    category: str
    description: str
    transaction_date: datetime


class TransactionCreate(TransactionBase):
    pass


class TransactionResponse(TransactionBase):
    id: UUID
    created_at: datetime

    class Config:
        from_attributes = True

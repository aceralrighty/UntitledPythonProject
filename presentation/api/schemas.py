from datetime import datetime
from decimal import Decimal
from uuid import UUID

from pydantic import BaseModel, EmailStr


class UserCreateRequest(BaseModel):
    username: str
    email: EmailStr
    password: str


class UserResponse(BaseModel):
    id: UUID
    username: str
    email: str
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True


class TransactionCreateRequest(BaseModel):
    account_id: UUID
    amount: Decimal
    category: str
    description: str
    transaction_date: datetime


class TransactionResponse(BaseModel):
    id: UUID
    account_id: UUID
    amount: Decimal
    category: str
    description: str
    transaction_date: datetime
    created_at: datetime

    class Config:
        from_attributes = True

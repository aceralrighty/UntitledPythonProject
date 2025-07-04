from dataclasses import dataclass
from datetime import datetime

from pydantic import BaseModel
from pydantic.v1 import UUID4
from sqlalchemy import DECIMAL


@dataclass
class TransactionCreate(BaseModel):
    account_id: UUID4
    amount: DECIMAL[10, 2]
    category: str
    description: str
    transaction_date: datetime


@dataclass
class TransactionResponse(BaseModel):
    id: UUID4
    amount: DECIMAL[10, 2]
    category: str
    created_at: datetime

    class Config:
        from_attributes = True


@dataclass
class User(BaseModel):
    id: UUID4
    username: str
    password: str
    email: str
    is_active: bool = True
    is_superuser: bool = False
    created_at: datetime = datetime.now()

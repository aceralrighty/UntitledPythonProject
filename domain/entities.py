from dataclasses import dataclass
from typing import Optional
from datetime import datetime
from decimal import Decimal


@dataclass
class User:
    username: str
    email: str
    password: str
    is_active: bool = True
    is_superuser: bool = False
    created_at: Optional[datetime] = None
    id: Optional[str] = None


@dataclass
class Transaction:
    account_id: str
    amount: Decimal
    category: str
    transaction_date: datetime
    description: Optional[str] = None
    created_at: Optional[datetime] = None
    id: Optional[str] = None

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional


class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Task:
    id: Optional[int] = None
    task_type: str = "reverse_text"
    payload: str = ""
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[str] = None
    created_at: datetime = datetime.now()
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    def mark_started(self):
        self.status = TaskStatus.IN_PROGRESS
        self.started_at = datetime.now()

    def mark_completed(self, result: str):
        self.status = TaskStatus.COMPLETED
        self.result = result
        self.completed_at = datetime.now()

    def mark_failed(self, error: str = ""):
        self.status = TaskStatus.FAILED
        self.result = error
        self.completed_at = datetime.now()


@dataclass
class User:
    id: Optional[int] = None
    username: str = ""
    password: str = ""

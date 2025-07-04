from pydantic import BaseModel
from enum import Enum


class TaskType(str, Enum):
    reverse = "reverse_text"
    compute = "compute_fib"


class TaskCreateRequest(BaseModel):
    task_type: TaskType
    payload: str


class TaskResponse(BaseModel):
    id: int
    status: str
    result: str | None = None

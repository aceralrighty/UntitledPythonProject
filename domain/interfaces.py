from typing import Protocol, List
from domain.entities import Task


class TaskRepository(Protocol):
    async def save(self: task: Task) -> Task:

    async def get_by_id(self, id: int) -> Task:


    async def list_pending_tasks(self) -> List[Task]:


    async def update(self, task: Task) -> Task:

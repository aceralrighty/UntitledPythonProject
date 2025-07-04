from fastapi import APIRouter, Depends, HTTPException
from presentation.api.schemas import TaskCreateRequest, TaskResponse
from application.services.task_service import TaskService
from infrastructure.db import SQLTaskRepository
from infrastructure.db.session import get_session

router = APIRouter()


@router.post("/", response_model=TaskResponse)
async def create_task(
        task: TaskCreateRequest,
        service: TaskService = Depends(TaskService)
):
    return await service.create_task(task)

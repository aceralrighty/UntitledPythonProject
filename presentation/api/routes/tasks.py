# app/api/routes/tasks.py
from fastapi import APIRouter, Depends, HTTPException
from app.api.schemas import TaskCreateRequest, TaskResponse
from app.application.services.task_service import TaskService
from app.infrastructure.db.repositories import SQLTaskRepository
from app.infrastructure.db.session import get_session

router = APIRouter()

@router.post("/", response_model=TaskResponse)
async def create_task(
    task: TaskCreateRequest,
    service: TaskService = Depends(TaskService)
):
    return await service.create_task(task)

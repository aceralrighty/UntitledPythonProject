# app/main.py
from fastapi import FastAPI
from app.api.routes import tasks

app = FastAPI(title="Async Task Service")

app.include_router(tasks.router, prefix="/tasks", tags=["Tasks"])

# app/main.py
from fastapi import FastAPI
from presentation.api.routes import tasks

app = FastAPI(title="Async Task Service")

app.include_router(tasks.router, prefix="/tasks", tags=["Tasks"])

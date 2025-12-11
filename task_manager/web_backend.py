import asyncio
import json
import uuid
import math
import random
import sys
import os
from typing import Any, Dict, List, Optional

# --- НОВІ ІМПОРТИ ДЛЯ РОЗДАЧІ ФАЙЛІВ ---
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import database

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError, model_validator
import redis.asyncio as redis

REDIS_URL = "redis://localhost:6379"
QUEUE_NAME = "tasks_queue"

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
r = redis.from_url(REDIS_URL, decode_responses=True)

# --- MODELS ---
class TSPInputData(BaseModel):
    cities_count: int = Field(..., gt=1, le=300)
    distance_matrix: Optional[List[List[float]]] = None 
    generate_random: bool = Field(False)
    coordinates: Optional[List[Dict[str, float]]] = None

    @model_validator(mode='after')
    def validate_logic(self):
        if not self.generate_random and not self.distance_matrix and not self.coordinates:
            raise ValueError("Required: generate_random=True OR coordinates list")
        return self

class CreateTaskRequest(BaseModel):
    type: str
    data: Dict[str, Any]

TASK_REGISTRY = {"tsp": TSPInputData}

# --- HELPERS ---
def generate_tsp_data(n: int, max_coord: float = 100.0):
    return [{"x": round(random.uniform(0, max_coord), 2), 
             "y": round(random.uniform(0, max_coord), 2)} for _ in range(n)]

# --- WEB ENDPOINT (ГОЛОВНА СТОРІНКА) ---
# Тепер при заході на http://localhost:8000/ відкриється ваш інтерфейс
@app.get("/")
async def read_index():
    # Шукаємо index.html у тій же папці, де лежить цей скрипт
    current_dir = os.path.dirname(os.path.abspath(__file__))
    index_path = os.path.join(current_dir, "index.html")
    return FileResponse(index_path)

# --- API ENDPOINTS ---
@app.get("/api/history")
async def get_history():
    return database.get_all_tasks()

@app.delete("/api/delete/{task_id}")
async def delete_task_endpoint(task_id: str):
    database.delete_task(task_id)
    return {"status": "ok", "deleted": task_id}

@app.post("/api/create")
async def create_task(request: CreateTaskRequest):
    if request.type not in TASK_REGISTRY:
        raise HTTPException(400, "Unknown task type")

    try:
        model_class = TASK_REGISTRY[request.type]
        input_model = model_class(**request.data)
    except ValidationError as e:
        raise HTTPException(422, e.errors())

    if isinstance(input_model, TSPInputData):
        if input_model.generate_random:
            input_model.coordinates = generate_tsp_data(input_model.cities_count)

    task_id = str(uuid.uuid4())
    full_input_data = input_model.model_dump(include={"cities_count", "coordinates"})
    
    database.add_task(task_id, request.type, full_input_data)

    service_payload = {
        "id": task_id,
        "type": request.type,
        "data": full_input_data
    }
    
    async with r.pipeline() as pipe:
        await pipe.lpush(QUEUE_NAME, json.dumps(service_payload))
        await pipe.execute()

    return {"task_id": task_id, "status": 0}

@app.get("/api/result/{task_id}")
async def get_full_result(task_id: str):
    task = database.get_task(task_id)
    if not task: raise HTTPException(404, "Task not found")
    
    return {
        "status": task['status'],
        "result": json.loads(task['result_data']) if task['result_data'] else None,
        "input_data": json.loads(task['input_data']) if task['input_data'] else None
    }

if __name__ == "__main__":
    import uvicorn
    # host="0.0.0.0" дозволяє доступ з інших комп'ютерів у мережі
    uvicorn.run(app, host="0.0.0.0", port=8000)

import asyncio
import json
import uuid
from typing import Any, Dict, List, Type

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError, validator
import redis.asyncio as redis

# --- CONFIG ---
REDIS_URL = "redis://localhost:6379"
QUEUE_NAME = "tasks_queue"
RESULT_KEY_PREFIX = "task_result:"
INPUT_KEY_PREFIX = "task_input:" # Зберігаємо вхідні дані, щоб повернути їх фронту разом з результатом

app = FastAPI(title="Modeling Service API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

r = redis.from_url(REDIS_URL, decode_responses=True)

# --- MODELS & REGISTRY ---

class TSPInputData(BaseModel):
    cities_count: int = Field(..., gt=1)
    distance_matrix: List[List[float]] = Field(...)

    @validator('distance_matrix')
    def validate_matrix(cls, v, values):
        count = values.get('cities_count')
        if count and (len(v) != count or any(len(row) != count for row in v)):
            raise ValueError("Matrix dimensions must match cities_count")
        return v

# Реєстр для розширення (можна додавати інші типи задач)
TASK_REGISTRY: Dict[str, Type[BaseModel]] = {
    "TSP_GENETIC": TSPInputData,
}

# --- API MODELS ---

class CreateTaskRequest(BaseModel):
    type: str
    data: Dict[str, Any]

class TaskResponse(BaseModel):
    task_id: str
    status: str

# --- ENDPOINTS ---

@app.get("/api/types")
async def get_types():
    return {"types": list(TASK_REGISTRY.keys())}

@app.post("/api/create", response_model=TaskResponse)
async def create_task(request: CreateTaskRequest):
    if request.type not in TASK_REGISTRY:
        raise HTTPException(400, "Unknown task type")

    try:
        validated_data = TASK_REGISTRY[request.type](**request.data)
    except ValidationError as e:
        raise HTTPException(422, e.errors())

    task_id = str(uuid.uuid4())
    
    # Дані для Windows Service (app.py)
    service_payload = {
        "id": task_id,
        "type": request.type,
        "data": validated_data.model_dump()
    }

    initial_status = {"id": task_id, "status": 0, "result": None}

    async with r.pipeline() as pipe:
        # 1. Зберігаємо статус
        await pipe.set(f"{RESULT_KEY_PREFIX}{task_id}", json.dumps(initial_status), ex=3600)
        # 2. Зберігаємо вхідні дані (щоб фронт міг намалювати графік, знаючи умови)
        await pipe.set(f"{INPUT_KEY_PREFIX}{task_id}", json.dumps(validated_data.model_dump()), ex=3600)
        # 3. Відправляємо в чергу
        await pipe.lpush(QUEUE_NAME, json.dumps(service_payload))
        await pipe.execute()

    return TaskResponse(task_id=task_id, status="queued")

@app.get("/api/result/{task_id}")
async def get_full_result(task_id: str):
    """
    Повертає повний пакет даних для фронтенду:
    INPUT (матриця) + OUTPUT (маршрут)
    """
    res_raw = await r.get(f"{RESULT_KEY_PREFIX}{task_id}")
    inp_raw = await r.get(f"{INPUT_KEY_PREFIX}{task_id}")

    if not res_raw:
        raise HTTPException(404, "Task not found")

    res_data = json.loads(res_raw)
    
    # Якщо задача готова, додаємо вхідні дані, щоб JS міг побудувати візуалізацію
    response_data = {
        "status": res_data["status"],
        "result": res_data["result"],
        "input_data": json.loads(inp_raw) if inp_raw else None
    }
    
    return response_data

# --- WEBSOCKET ---

@app.websocket("/ws/{task_id}")
async def websocket_endpoint(websocket: WebSocket, task_id: str):
    await websocket.accept()
    last_status = -1
    
    try:
        while True:
            raw = await r.get(f"{RESULT_KEY_PREFIX}{task_id}")
            if raw:
                data = json.loads(raw)
                curr = data.get("status")

                if curr != last_status:
                    await websocket.send_json(data)
                    last_status = curr
                
                if curr in [2, 3]:
                    await asyncio.sleep(0.2)
                    await websocket.close()
                    break
            else:
                await websocket.send_json({"error": "Not found"})
                await websocket.close()
                break
            
            await asyncio.sleep(0.5)
            
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WS Error: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

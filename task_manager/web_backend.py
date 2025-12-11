import asyncio
import json
import uuid
import math
import random
from typing import Any, Dict, List, Optional, Type, Tuple

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError, model_validator
import redis.asyncio as redis

# --- 1. КОНФІГУРАЦІЯ ---
REDIS_URL = "redis://localhost:6379"
QUEUE_NAME = "tasks_queue"
RESULT_KEY_PREFIX = "task_result:"
INPUT_KEY_PREFIX = "task_input:"

app = FastAPI(title="Modeling Service API")

# Дозвіл для фронтенду
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

r = redis.from_url(REDIS_URL, decode_responses=True)

# --- 2. ЛОГІКА ГЕНЕРАЦІЇ (Helper Function) ---
def generate_tsp_data(n: int, max_coord: float = 100.0) -> Tuple[List[Dict[str, float]], List[List[float]]]:
    """Генерує випадкові координати та матрицю відстаней для N міст."""
    # Генеруємо точки (для графіків)
    points = [{"x": round(random.uniform(0, max_coord), 2), 
               "y": round(random.uniform(0, max_coord), 2)} for _ in range(n)]
    
    # Генеруємо матрицю (для алгоритму)
    matrix = []
    for i in range(n):
        row = []
        for j in range(n):
            if i == j:
                dist = 0.0
            else:
                p1, p2 = points[i], points[j]
                dist = math.sqrt((p1["x"] - p2["x"])**2 + (p1["y"] - p2["y"])**2)
            row.append(round(dist, 2))
        matrix.append(row)
    return points, matrix

# --- 3. МОДЕЛІ ДАНИХ ТА ВАЛІДАЦІЯ ---

class TSPInputData(BaseModel):
    cities_count: int = Field(..., gt=1, le=100, description="Кількість міст")
    
    # Матриця може бути None, ТІЛЬКИ якщо generate_random=True
    distance_matrix: Optional[List[List[float]]] = None 
    generate_random: bool = Field(False, description="Авто-генерація міст")
    
    # Поле для збереження координат (якщо згенеровано або введено вручну, якщо розширити логіку)
    coordinates: Optional[List[Dict[str, float]]] = None

    @model_validator(mode='after')
    def validate_logic(self):
        """Головна перевірка: Або матриця вручну, Або генерація"""
        if not self.generate_random and not self.distance_matrix:
            raise ValueError("Необхідно ввести 'distance_matrix' АБО встановити 'generate_random=True'")
        
        # Якщо користувач ввів матрицю вручну - перевіряємо її розмір
        if self.distance_matrix:
            if len(self.distance_matrix) != self.cities_count:
                raise ValueError(f"Розмір матриці ({len(self.distance_matrix)}) не співпадає з кількістю міст ({self.cities_count})")
            for row in self.distance_matrix:
                if len(row) != self.cities_count:
                    raise ValueError("Матриця має бути квадратною (N x N)")
        
        return self

    def model_json(self):
        return self.model_dump_json()

# Реєстр задач (Scaleable part)
TASK_REGISTRY: Dict[str, Type[BaseModel]] = {
    "TSP_GENETIC": TSPInputData,
}

# --- 4. API ЕНДПОІНТИ ---

class CreateTaskRequest(BaseModel):
    type: str
    data: Dict[str, Any]

class TaskResponse(BaseModel):
    task_id: str
    status: int
    message: str

@app.get("/api/types")
async def get_types():
    return {"types": list(TASK_REGISTRY.keys())}

@app.post("/api/create", response_model=TaskResponse)
async def create_task(request: CreateTaskRequest):
    # 1. Перевірка типу задачі
    if request.type not in TASK_REGISTRY:
        raise HTTPException(400, "Unknown task type")

    # 2. Валідація вхідних даних (Pydantic)
    try:
        model_class = TASK_REGISTRY[request.type]
        input_model = model_class(**request.data)
    except ValidationError as e:
        raise HTTPException(422, e.errors())

    # 3. Обробка генерації (якщо обрано авто-режим)
    if isinstance(input_model, TSPInputData) and input_model.generate_random:
        coords, matrix = generate_tsp_data(input_model.cities_count)
        input_model.distance_matrix = matrix
        input_model.coordinates = coords # Зберігаємо для фронтенду

    # 4. Підготовка даних для Windows Service
    task_id = str(uuid.uuid4())
    
    # app.py отримає тільки те, що йому треба (без зайвих полів типу generate_random)
    service_payload = {
        "id": task_id,
        "type": request.type,
        "data": input_model.model_dump(include={"cities_count", "distance_matrix"})
    }

    initial_status = {"id": task_id, "status": 0, "result": None}

    # 5. Запис у Redis (Атомарно)
    async with r.pipeline() as pipe:
        # Статус для моніторингу
        await pipe.set(f"{RESULT_KEY_PREFIX}{task_id}", json.dumps(initial_status), ex=3600)
        # Повні вхідні дані (включно з координатами) для малювання графіка на фронті
        await pipe.set(f"{INPUT_KEY_PREFIX}{task_id}", input_model.model_json(), ex=3600)
        # Задача в чергу для app.py
        await pipe.lpush(QUEUE_NAME, json.dumps(service_payload))
        await pipe.execute()

    msg = "Task generated automatically" if input_model.generate_random else "Task created from manual input"
    return TaskResponse(task_id=task_id, status=0, message=msg)

@app.get("/api/result/{task_id}")
async def get_full_result(task_id: str):
    """Повертає результат + вхідні дані (щоб фронтенд міг намалювати графік)"""
    res_raw = await r.get(f"{RESULT_KEY_PREFIX}{task_id}")
    inp_raw = await r.get(f"{INPUT_KEY_PREFIX}{task_id}")

    if not res_raw:
        raise HTTPException(404, "Task not found")

    res_data = json.loads(res_raw)
    
    response_data = {
        "status": res_data["status"],
        "result": res_data["result"],
        "input_data": json.loads(inp_raw) if inp_raw else None
    }
    return response_data

# --- 5. REAL-TIME WEBSOCKET ---

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
                
                if curr in [2, 3]: # Done or Error
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

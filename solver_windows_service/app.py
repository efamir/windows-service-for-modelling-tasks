import asyncio
import json
import logging
import os
import sys
import random
import traceback

# --- FIX PATH: АВТОМАТИЧНЕ ДОДАВАННЯ ШЛЯХУ ---
# Це критично для запуску як Служби Windows (бо вона стартує з system32)
# Ми знаходимо папку, де лежить цей скрипт, і додаємо батьківську папку в sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# ---------------------------------------------

# --- FIX GRAPHICS ---
import matplotlib
matplotlib.use('Agg') 

import database # Тепер імпорт спрацює гарантовано

import win32serviceutil
import win32service
import win32event
import servicemanager
import redis.asyncio as redis

from solvers_config import task_name_executor_map

# --- LOGGING ---
# Важливо: використовуємо абсолютний шлях для логів
LOG_FILE = os.path.join(current_dir, 'service_log.txt')

file_handler = logging.FileHandler(LOG_FILE, 'a', 'utf-8')
# Console handler прибрали, бо у служби немає консолі
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S',
    handlers=[file_handler]
)

class SolverService(win32serviceutil.ServiceFramework):
    _svc_name_ = "SolverService"
    _svc_display_name_ = "Python Solver Service (TSP)" # ЦЕ ІМ'Я ВИ ПОБАЧИТЕ В WINDOWS
    _svc_description_ = "Виконує важкі математичні розрахунки для проекту ТВ-33."

    def __init__(self, args):
        win32serviceutil.ServiceFramework.__init__(self, args)
        self.hWaitStop = win32event.CreateEvent(None, 0, 0, None)
        self.running = True

    def SvcStop(self):
        logging.info("!!! STOP SIGNAL RECEIVED !!!")
        self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
        win32event.SetEvent(self.hWaitStop)
        self.running = False

    def SvcDoRun(self):
        logging.info("=== SERVICE STARTED BY WINDOWS ===")
        try:
            asyncio.run(self.main())
        except Exception as e:
            logging.critical(f"CRASH: {e}")
            logging.critical(traceback.format_exc())
            self.SvcStop()

    async def main(self):
        logging.info("Connecting to Redis...")
        try:
            r = redis.from_url("redis://localhost:6379", decode_responses=True)
            await r.ping()
            logging.info("Redis connected.")
        except Exception as e:
            logging.critical(f"REDIS FAILED: {e}")
            return

        sem = asyncio.Semaphore(4)

        # RECOVERY
        try:
            pending = database.get_pending_or_processing_tasks()
            if pending:
                logging.info(f"[RECOVERY] Re-queuing {len(pending)} tasks...")
                for task in pending:
                    input_d = json.loads(task['input_data'])
                    await r.lpush("tasks_queue", json.dumps({"id": task['id'], "type": task['type'], "data": input_d}))
        except Exception as e:
            logging.error(f"Recovery failed: {e}")

        while self.running:
            if win32event.WaitForSingleObject(self.hWaitStop, 0) == win32event.WAIT_OBJECT_0:
                break
            try:
                item = await r.brpop("tasks_queue", timeout=1)
                if item:
                    asyncio.create_task(self.process_task(r, item[1], sem))
            except redis.ConnectionError:
                await asyncio.sleep(5)
            except Exception:
                await asyncio.sleep(1)

        await r.aclose()
        logging.info("Service stopped.")

    async def process_task(self, r_client, task_json, sem):
        try:
            data = json.loads(task_json)
            t_id = data.get("id")
            async with sem:
                logging.info(f"START {t_id}")
                database.update_task_status(t_id, 1)
                await r_client.publish(f"task_update:{t_id}", json.dumps({"status": 1}))

                executor = task_name_executor_map.get(data.get("type"))
                
                # Data Prep
                t_input = data.get("data", {})
                if "coordinates" in t_input:
                    points = [[c['x'], c['y']] for c in t_input["coordinates"]]
                else:
                    points = [[random.randint(0,100), random.randint(0,100)] for _ in range(t_input.get("cities_count", 10))]

                res_raw = await executor.solve(json.dumps(points))
                result = json.loads(res_raw)

                database.update_task_status(t_id, 2, result)
                await r_client.publish(f"task_update:{t_id}", json.dumps({"status": 2, "result": result}))
                logging.info(f"DONE {t_id}")

        except Exception as e:
            logging.error(f"ERROR: {e}")
            if 't_id' in locals():
                database.update_task_status(t_id, 3, {"error": str(e)})
                await r_client.publish(f"task_update:{t_id}", json.dumps({"status": 3}))

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    if len(sys.argv) == 1:
        servicemanager.Initialize()
        servicemanager.PrepareToHostSingle(SolverService)
        servicemanager.StartServiceCtrlDispatcher()
    else:
        win32serviceutil.HandleCommandLine(SolverService)

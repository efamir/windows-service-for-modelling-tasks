import asyncio
import json
import logging
import os
import sys
import socket

# Бібліотеки для Windows Service
import win32serviceutil
import win32service
import win32event
import servicemanager

# Клієнт Redis
import redis.asyncio as redis

# Імпорти проекту
# Важливо: matplotlib може конфліктувати в сервісі без бекенда Agg
import matplotlib
matplotlib.use('Agg') 

from solvers_config import task_name_executor_map

# Налаштування логування (бо print у службі не видно)
LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'service_log.txt')
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class SolverService(win32serviceutil.ServiceFramework):
    _svc_name_ = "SolverService"
    _svc_display_name_ = "Python Solver Service (TSP)"
    _svc_description_ = "Processes algorithmic tasks via Redis queue using AsyncIO and Multiprocessing."

    def __init__(self, args):
        win32serviceutil.ServiceFramework.__init__(self, args)
        self.hWaitStop = win32event.CreateEvent(None, 0, 0, None)
        self.running = True
        self.loop = None

    def SvcStop(self):
        """Викликається, коли Windows зупиняє службу"""
        logging.info("Service is stopping...")
        self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
        win32event.SetEvent(self.hWaitStop)
        self.running = False

    def SvcDoRun(self):
        """Основна точка входу служби"""
        logging.info("Service is starting...")
        servicemanager.LogMsg(
            servicemanager.EVENTLOG_INFORMATION_TYPE,
            servicemanager.PYS_SERVICE_STARTED,
            (self._svc_name_, '')
        )
        
        # Запуск асинхронного циклу
        try:
            asyncio.run(self.main())
        except Exception as e:
            logging.error(f"Critical error in main loop: {e}")
            self.SvcStop()

    async def main(self):
        """Головний цикл обробки Redis"""
        # Підключення до Redis (налаштуйте хост/порт під себе)
        r = redis.from_url("redis://localhost:6379", decode_responses=True)
        
        # Семафор для обмеження кількості одночасних задач (макс 4)
        sem = asyncio.Semaphore(4)
        
        logging.info("Connected to Redis. Waiting for tasks...")

        while self.running:
            # Перевіряємо сигнал зупинки Windows (неблокуючим способом)
            rc = win32event.WaitForSingleObject(self.hWaitStop, 0)
            if rc == win32event.WAIT_OBJECT_0:
                break

            try:
                # brpop блокує виконання, доки не з'явиться елемент або не вийде таймаут (1 сек)
                # Це дозволяє перевіряти self.running кожну секунду
                item = await r.brpop("tasks_queue", timeout=1)
                
                if item:
                    # item повертає кортеж ('tasks_queue', 'json_str')
                    _, task_json = item
                    # Створюємо задачу у фоні, не блокуючи цикл
                    asyncio.create_task(self.process_task(r, task_json, sem))
            
            except redis.ConnectionError:
                logging.error("Redis connection lost. Retrying in 5 seconds...")
                await asyncio.sleep(5)
            except Exception as e:
                logging.error(f"Error in loop: {e}")
                await asyncio.sleep(1)

        logging.info("Service loop finished.")
        await r.aclose()

    async def process_task(self, r_client, task_json, sem):
        """Обробка окремої задачі"""
        task_data = {}
        try:
            task_data = json.loads(task_json)
            t_id = task_data.get("id")
            t_type = task_data.get("type")
            t_input = task_data.get("data")
            
            result_key = f"task_result:{t_id}"

            # 1. Захоплюємо слот семафору (якщо 4 зайняті, чекаємо тут)
            async with sem:
                logging.info(f"Processing task {t_id} of type {t_type}")
                
                # STATUS: 1 (Processing)
                await self.update_status(r_client, result_key, t_id, None, 1)

                executor_class = task_name_executor_map.get(t_type)
                
                if not executor_class:
                    raise ValueError(f"Unknown task type: {t_type}")

                # Виконуємо задачу (вона використовує ProcessPoolExecutor всередині)
                # result повертається як JSON рядок з core.py
                result_raw = await executor_class.solve(str(t_input))
                result = json.loads(result_raw)

                # STATUS: 2 (Done)
                await self.update_status(r_client, result_key, t_id, result, 2)
                logging.info(f"Task {t_id} completed successfully.")

        except Exception as e:
            logging.error(f"Error processing task: {e}")
            t_id = task_data.get("id", "unknown")
            result_key = f"task_result:{t_id}"
            
            # STATUS: 3 (Error)
            error_msg = str(e)
            await self.update_status(r_client, result_key, t_id, error_msg, 3)

    async def update_status(self, r_client, key, t_id, result, status):
        """Оновлює статус задачі в Redis"""
        response = {
            "id": t_id,
            "result": result,
            "status": status
        }
        # Зберігаємо результат як JSON рядок за ключем ID
        # Можна використати HSET, якщо потрібно зберігати всі результати в одній хеш-таблиці
        await r_client.set(key, json.dumps(response))
        # Опціонально: ставимо TTL, щоб результати не висіли вічно
        await r_client.expire(key, 3600) 

if __name__ == '__main__':
    # ВАЖЛИВО: Для multiprocessing в Windows
    import multiprocessing
    multiprocessing.freeze_support()

    if len(sys.argv) == 1:
        servicemanager.Initialize()
        servicemanager.PrepareToHostSingle(SolverService)
        servicemanager.StartServiceCtrlDispatcher()
    else:
        win32serviceutil.HandleCommandLine(SolverService)

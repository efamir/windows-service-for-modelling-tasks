# Цей файл є демонстрацією використання TSPSolver з його багатопроцесорністю
# В майбутньому тут повинен бути слухач на оновлення redis


import asyncio
import json
import random
import time

from tsp_solver import TSPSolver


def create_cities():
    return [[random.randint(-100, 100), random.randint(-100, 100)] for _ in range(100)]


def create_cities_float():
    def get_random_float(from_: int, to: int):
        return (to - from_) * random.random() + from_

    return [(get_random_float(-100, 100), get_random_float(-100, 100)) for _ in range(100)]


async def main():
    tasks = [asyncio.create_task(TSPSolver.solve(str(create_cities()))) for _ in range(4)]
    reses = await asyncio.gather(*tasks)
    for res in reses:
        res = json.loads(res)
        res["best_root_graph"] = res["best_root_graph"][:10] + "... (rest is hidden for the showcase)"
        print(res)


if __name__ == '__main__':
    start = time.perf_counter()
    print("Started")
    asyncio.run(main())
    print(f"{time.perf_counter() - start:.4f} secs")

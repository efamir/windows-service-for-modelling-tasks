import asyncio
from tsp_solver import TSPSolver
import time
import random


def create_cities():
    return [[random.randint(-100, 100), random.randint(-100, 100)] for _ in range(40)]


def create_cities_float():
    def get_random_float(from_: int, to: int):
        return (to - from_) * random.random() + from_

    return [(get_random_float(-100, 200), get_random_float(-110, 150)) for _ in range(40)]


async def main():
    tasks = [asyncio.create_task(TSPSolver.solve(str(create_cities()))) for _ in range(4)]
    reses = await asyncio.gather(*tasks)
    for res in reses:
        print(res)


if __name__ == '__main__':
    start = time.perf_counter()
    print("Starting async")
    asyncio.run(main())
    print(f"{time.perf_counter() - start:.4f} secs")

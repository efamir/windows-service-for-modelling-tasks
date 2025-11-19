import asyncio
from tsp_solver import TSPSolver
import time
import random


def create_cities():
    return [(random.randint(-100, 100), random.randint(-100, 100)) for _ in range(40)]


async def main(cities: list[tuple[int, int]]):
    tasks = [asyncio.create_task(TSPSolver.solve(create_cities())) for _ in range(4)]
    await asyncio.gather(*tasks)


def sync(cities: list[tuple[int, int]]):
    for _ in range(4):
        tsp = TSPSolver(cities)
        cities_graph = tsp.plot_cities_graph()
        tsp._solve()
        shortest_distance = tsp.best_overall[1]
        print("Shortest distance: ", shortest_distance)
        solving_progression = tsp.plot_solving_progression(avg=True)
        best_root = tsp.plot_best_route()


if __name__ == '__main__':
    cities = [(random.randint(-100, 100), random.randint(-100, 100)) for _ in range(10)]

    start = time.perf_counter()
    print("Starting async")
    asyncio.run(main(cities))
    print(f"{time.perf_counter() - start:.4f} secs")

    # start = time.perf_counter()
    # print("Starting sync")
    # sync(cities)
    # print(time.perf_counter() - start)

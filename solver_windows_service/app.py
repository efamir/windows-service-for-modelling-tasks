import asyncio
from tsp_solver import TSPSolver
import time
import random


def create_cities():
    return [(random.randint(-100, 100), random.randint(-100, 100)) for _ in range(40)]


def create_cities_float():
    def get_random_float(from_: int, to: int):
        return (to - from_) * random.random() + from_

    return [(get_random_float(-100, 200), get_random_float(-110, 150)) for _ in range(40)]


async def main(cities: list[tuple[int, int]]):
    tasks = [asyncio.create_task(TSPSolver.solve(create_cities_float())) for _ in range(1)]
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
    cities = [(random.randint(-100, 200), random.randint(-100, 150)) for _ in range(10)]

    start = time.perf_counter()
    print("Starting async")
    asyncio.run(main(cities))
    print(f"{time.perf_counter() - start:.4f} secs")

    # start = time.perf_counter()
    # print("Starting sync")
    # sync(cities)
    # print(time.perf_counter() - start)

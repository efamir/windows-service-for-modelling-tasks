import asyncio
import base64
import io
import os
import random
import uuid

import matplotlib.pyplot as plt
import numpy as np
from deap import base, creator, tools, algorithms

from solver_windows_service.solver import AsyncSolver, get_shared_executor

MAX_X = 100
MAX_Y = 100
MAX_WORKERS = os.cpu_count()


class City:
    def __init__(self, x, y):
        self.x = x
        self.y = y


if not hasattr(creator, "FitnessMin"):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
if not hasattr(creator, "Route"):
    creator.create("Route", list, fitness=creator.FitnessMin)


class TSPSolver(AsyncSolver):
    @staticmethod
    def calculate_distance(city1: City, city2: City):
        return ((city1.x - city2.x) ** 2 + (city1.y - city2.y) ** 2) ** 0.5

    # Функція оцінки придатності
    def __evaluate(self, route):
        if len(route) < 2:
            raise AttributeError("Route have to have at least two __cities to be evaluated")
        total_distance = 0
        for i in range(1, len(route)):
            total_distance += self.__distances[route[i - 1]][route[i]]
        total_distance += self.__distances[route[-1]][route[0]]
        return (total_distance,)

    def __indices(self):
        if random.random() < (1 - self.__p_qualitative):
            return random.sample(range(self.__cities_count), self.__cities_count)

        start = random.randint(0, self.__cities_count - 1)
        result = [start]
        count = 1
        while count < self.__cities_count:
            from_ = result[-1]
            min_dist = 99999999999
            min_ind = -1
            for i in range(self.__cities_count):
                if from_ == i:
                    continue
                distance = self.__distances[from_][i]
                if i not in result and distance < min_dist:
                    min_dist = distance
                    min_ind = i

            if min_ind == -1:
                raise ValueError("CITIES_COUNT is wrong")

            result.append(min_ind)
            count += 1

        return result

    def __init__(self, cities: list[tuple[float, float]] = None,
                 cities_count=100, population_size=300, max_generations=200,
                 p_crossover=0.09, p_mutation=0.9,
                 tournsize=7, p_qualitative=0.6):
        if not 2 <= (cities_count if not cities else len(cities)) <= 300:
            raise ValueError("cities_count must be between 2 and 300")
        if not 1 <= population_size <= 1000:
            raise ValueError("population_size must be between 1 and 1000")
        if not 1 <= max_generations <= 1000:
            raise ValueError("max_generations must be between 1 and 1000")
        if not 0 <= p_crossover <= 1:
            raise ValueError("p_crossover must be between 0 and 1")
        if not 0 <= p_mutation <= 1:
            raise ValueError("p_mutation must be between 0 and 1")
        if not 0 <= p_qualitative <= 1:
            raise ValueError("p_qualitative must be between 0 and 1")
        if p_crossover + p_mutation > 1:
            raise ValueError("p_crossover and p_mutation sum cannot be greater than 1")
        if not 2 <= tournsize <= cities_count:
            raise ValueError("tournsize must be between 2 and cities_count")

        self.__logbook = None
        self.__cities_count = cities_count if not cities else len(cities)
        self.__population_size = population_size
        self.__max_generations = max_generations
        self.__p_crossover = p_crossover
        self.__p_mutation = p_mutation
        self.__lambda = self.__population_size
        self.__p_qualitative = p_qualitative

        self.__cities = []
        self.__cities_xs = []
        self.__cities_ys = []
        if not cities:
            for _ in range(self.__cities_count):
                new_city = City(random.randint(0, MAX_X), random.randint(0, MAX_Y))
                self.__cities.append(new_city)
                self.__cities_xs.append(new_city.x)
                self.__cities_ys.append(new_city.y)
        else:
            for x, y in cities:
                new_city = City(x, y)
                self.__cities.append(new_city)
                self.__cities_xs.append(new_city.x)
                self.__cities_ys.append(new_city.y)

        self.__distances = []
        for i in range(len(self.__cities)):
            new_row = []
            for j in range(len(self.__cities)):
                new_row.append(TSPSolver.calculate_distance(self.__cities[i], self.__cities[j]))
            self.__distances.append(new_row)

        self.__toolbox = base.Toolbox()
        # Cхрещування
        self.__toolbox.register("mate", tools.cxOrdered)
        # Мутація
        self.__toolbox.register("mutate", tools.mutInversion)
        # Вибірка для наступного покоління
        self.__toolbox.register("select", tools.selTournament, tournsize=tournsize)
        self.__toolbox.register("evaluate", self.__evaluate)
        self.__toolbox.register("indices", random.sample, range(self.__cities_count), self.__cities_count)
        self.__toolbox.register("individual", tools.initIterate, creator.Route, self.__indices)
        self.__toolbox.register("population", tools.initRepeat, list, self.__toolbox.individual)

        self.__stats = tools.Statistics(lambda route: route.fitness.values)
        self.__stats.register("min", np.min)
        self.__stats.register("avg", np.mean)

        self.__best_overall = None

    def plot_cities_graph(self):
        plt.figure(figsize=(8, 8))
        plt.scatter(self.__cities_xs, self.__cities_ys)
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png")
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close()
        return image_base64

    def _solve(self):
        # Ініціалізація популяції
        pop = self.__toolbox.population(n=self.__cities_count)
        hof = tools.HallOfFame(1)

        population, self.__logbook = algorithms.eaMuPlusLambda(
            pop,
            self.__toolbox,
            mu=self.__population_size,
            lambda_=self.__lambda,
            cxpb=self.__p_crossover,
            mutpb=self.__p_mutation,
            ngen=self.__max_generations,
            stats=self.__stats,
            halloffame=hof,
            verbose=False
        )

        self.__best_overall = hof[0]

    @staticmethod
    async def solve(cities: list[tuple[float, float]] = None):
        loop = asyncio.get_running_loop()
        cities_len = len(cities)
        pop_size = cities_len * 3
        max_generations = cities_len * 2
        raw_tournsize = pop_size * 0.02333
        calculated_tournsize = int(round(raw_tournsize))
        tournsize = max(2, min(calculated_tournsize, pop_size))
        return await loop.run_in_executor(
            get_shared_executor(), TSPSolver._solve_for_executor, cities, pop_size, max_generations, tournsize
        )

    @staticmethod
    def _solve_for_executor(cities: list[tuple[float, float]] = None,
                            population_size=300,
                            max_generations=200,
                            tournsize=7):
        tsp = TSPSolver(cities, population_size=population_size, max_generations=max_generations, tournsize=tournsize)
        cities_graph = tsp.plot_cities_graph()
        tsp._solve()
        shortest_distance = tsp.best_overall[1]
        print("Shortest distance: ", shortest_distance)
        solving_progression = tsp.plot_solving_progression(avg=True)
        best_root = tsp.plot_best_route()
        return shortest_distance, tsp.best, best_root, cities_graph, solving_progression

    @property
    def best_overall(self):
        return self.__best_overall.copy(), self.__best_overall.fitness.values[0]

    def plot_solving_progression(self, avg=False):
        if not self.__best_overall:
            return

        min_fit, avg_fit = self.__logbook.select("min", "avg")

        plt.figure()
        plt.plot(min_fit, color="red")

        if avg:
            plt.plot(avg_fit, color="green")

        plt.xlabel("Generation")
        if avg:
            plt.ylabel("MIN/AVG fitness")
        else:
            plt.ylabel("MIN fitness")
        plt.title("Evolution with eaMuPlusLambda (Elitism)")
        plt.grid(True)
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png")
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close()
        return image_base64

    @property
    def best(self):
        if not self.__best_overall:
            return []

        return [(self.__cities[city].x, self.__cities[city].y) for city in self.__best_overall]

    def plot_best_route(self):
        if not self.__best_overall:
            return

        plt.figure(figsize=(8, 8))
        plt.scatter(self.__cities_xs, self.__cities_ys)
        for i in range(len(self.__best_overall)):
            city1 = self.__cities[self.__best_overall[i]]
            if i + 1 == self.__cities_count:
                city2 = self.__cities[self.__best_overall[0]]
            else:
                city2 = self.__cities[self.__best_overall[i + 1]]

            plt.plot([city1.x, city2.x], [city1.y, city2.y], "r-", linewidth=1)
        plt.tight_layout()

        # TODO: remove, because temp
        debug_dir = "debug_plots"
        os.makedirs(debug_dir, exist_ok=True)
        random_filename = f"{uuid.uuid4().hex}.png"
        save_path = os.path.join(debug_dir, random_filename)
        plt.savefig(save_path, format="png")
        print(f"!!! DEBUG: Best route file result: {save_path}")
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png")
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close()
        return image_base64

        # buffer = io.BytesIO()
        # plt.plot()
        # plt.savefig(buffer, format="png")
        # buffer.seek(0)
        # image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        # plt.close()
        # return image_base64

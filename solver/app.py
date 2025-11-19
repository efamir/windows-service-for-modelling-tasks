import asyncio
from tsp_solver import TSPSolver

if __name__ == '__main__':
    asyncio.run(TSPSolver.solve())
    # tsp = TSPSolver()
    # tsp.plot_cities_graph()
    # tsp._solve()
    # print("Shortest distance: ", tsp.best_overall[1])
    # tsp.plot_solving_progression(avg=True)
    # tsp.plot_best_route()

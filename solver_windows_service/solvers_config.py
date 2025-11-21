from typing import Type

from solver import Solver
from tsp_solver import TSPSolver

task_name_executor_map: dict[str, Type[Solver]] = {
    "tsp": TSPSolver,
}

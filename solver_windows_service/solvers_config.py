from solver import Solver
from typing import Type

from tsp_solver import TSPSolver

task_name_executor_map: dict[str, Type[Solver]] = {
    "tsp": TSPSolver,
}

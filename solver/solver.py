from abc import ABC, abstractmethod


class Solver(ABC):
    @staticmethod
    @abstractmethod
    def solve() -> dict:
        pass


class AsyncSolver(Solver):
    @staticmethod
    @abstractmethod
    async def solve():
        pass

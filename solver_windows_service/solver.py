import atexit
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor

MAX_WORKERS = 4
_SHARED_EXECUTOR = None


def get_shared_executor() -> ProcessPoolExecutor:
    global _SHARED_EXECUTOR
    if _SHARED_EXECUTOR is not None:
        return _SHARED_EXECUTOR

    _SHARED_EXECUTOR = ProcessPoolExecutor(max_workers=MAX_WORKERS)
    print("Pool executor created")
    atexit.register(cleanup_executor)

    return _SHARED_EXECUTOR


def cleanup_executor():
    global _SHARED_EXECUTOR
    if _SHARED_EXECUTOR:
        print("Shutting down ProcessPoolExecutor...")
        _SHARED_EXECUTOR.shutdown(wait=True)
        _SHARED_EXECUTOR = None
        print("ProcessPoolExecutor has been shut down")


class Solver(ABC):
    @staticmethod
    @abstractmethod
    def solve() -> str:
        pass


class AsyncSolver(Solver):
    @staticmethod
    @abstractmethod
    async def solve() -> str:
        pass

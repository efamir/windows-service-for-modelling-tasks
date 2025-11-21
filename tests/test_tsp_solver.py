import asyncio
import json
from unittest.mock import MagicMock, patch

import pytest

from solver_windows_service.tsp_solver.core import City, TSPSolver


# Блок 1: Тестування логіки (Unit Tests)
class TestCity:
    def test_city_init(self):
        """Перевірка коректної ініціалізації міста."""
        city = City(10, 20)
        assert city.x == 10
        assert city.y == 20


class TestTSPSolverInit:
    def test_init_defaults(self):
        """Перевірка ініціалізації з дефолтними значеннями."""
        solver = TSPSolver(cities_count=100)
        assert solver._TSPSolver__cities_count == 100
        assert len(solver._TSPSolver__cities) == 100

    def test_init_validation_errors(self):
        """Перевірка валідації вхідних даних."""
        with pytest.raises(ValueError, match="cities_count must be between"):
            TSPSolver(cities_count=1)

        with pytest.raises(ValueError, match="p_crossover and p_mutation sum"):
            TSPSolver(p_crossover=0.8, p_mutation=0.8)

        with pytest.raises(ValueError, match="tournsize must be between"):
            TSPSolver(cities_count=5, tournsize=6)

        with pytest.raises(ValueError, match="population_size must be between"):
            TSPSolver(population_size=10000)

        with pytest.raises(ValueError, match="max_generations must be between"):
            TSPSolver(max_generations=0)

        with pytest.raises(ValueError, match="p_crossover must be between"):
            TSPSolver(p_crossover=-0.1)

        with pytest.raises(ValueError, match="p_mutation must be between"):
            TSPSolver(p_mutation=1.1)

        with pytest.raises(ValueError, match="p_qualitative must be between"):
            TSPSolver(p_qualitative=1.5)

    def test_init_random_cities(self):
        """Тест генерації випадкових міст (коли cities=None)."""
        solver = TSPSolver(cities=None, cities_count=10)
        assert len(solver._TSPSolver__cities) == 10
        assert len(solver._TSPSolver__cities_xs) == 10

    def test_calculate_distance(self):
        """Перевірка розрахунку дистанції."""
        c1 = City(0, 0)
        c2 = City(3, 4)
        dist = TSPSolver.calculate_distance(c1, c2)
        assert dist == 5.0


# Блок 2: Допоміжні класи
class MockRoute(list):
    """Фейковий клас маршруту."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fitness = MagicMock()
        self.fitness.values = (123.45,)

    def copy(self):
        return list(self)


# Блок 3: Тестування з Mock-об'єктами
class TestTSPSolverLogic:

    @patch("solver_windows_service.tsp_solver.core.plt")
    def test_plot_cities_graph(self, mock_plt):
        """Тест генерації графіка міст."""
        solver = TSPSolver(cities_count=5, tournsize=2)
        mock_plt.savefig.return_value = None
        result_base64 = solver.plot_cities_graph()
        assert isinstance(result_base64, str)
        mock_plt.figure.assert_called_once()
        mock_plt.close.assert_called_once()

    @patch("solver_windows_service.tsp_solver.core.tools.HallOfFame")
    @patch("solver_windows_service.tsp_solver.core.algorithms.eaMuPlusLambda")
    def test_solve_internal_logic(self, mock_ea_algo, mock_hof_class):
        """Тест внутрішньої логіки розв'язання (DEAP)."""
        solver = TSPSolver(cities_count=5, max_generations=10, tournsize=2)
        fake_best_individual = MockRoute([0, 1, 2, 3, 4])

        mock_hof_instance = mock_hof_class.return_value
        mock_hof_instance.__getitem__.return_value = fake_best_individual

        mock_logbook = MagicMock()
        mock_ea_algo.return_value = ([fake_best_individual], mock_logbook)

        solver._solve()

        best_route, distance = solver.best_overall
        assert distance == 123.45
        assert best_route == [0, 1, 2, 3, 4]
        assert mock_ea_algo.called

    @patch("solver_windows_service.tsp_solver.core.get_shared_executor")
    @patch("asyncio.get_running_loop")
    @pytest.mark.asyncio
    async def test_solve_async_entry_point(self, mock_loop, mock_executor):
        """Тест асинхронної точки входу."""
        cities_data = [[0, 0], [10, 10]]
        cities_json = json.dumps(cities_data)

        expected_dict = {"status": "fake_ok"}
        future_result = asyncio.Future()
        future_result.set_result(expected_dict)

        mock_loop.return_value.run_in_executor.return_value = future_result

        result = await TSPSolver.solve(inp=cities_json)
        assert result == expected_dict
        mock_loop.return_value.run_in_executor.assert_called_once()

    @patch("solver_windows_service.tsp_solver.core.TSPSolver")
    def test_solve_for_executor(self, MockTSPSolverClass):
        """Тест методу для executor."""
        mock_instance = MockTSPSolverClass.return_value
        mock_instance.best_overall = ([0, 1], 50.0)
        mock_instance.best = [[0, 0], [10, 10]]
        mock_instance.plot_best_route.return_value = "base64string"

        result_json = TSPSolver._solve_for_executor(cities=[[0, 0], [10, 10]])
        result = json.loads(result_json)

        assert result["shortest_distance"] == 50.0
        assert result["best_root_graph"] == "base64string"
        mock_instance._solve.assert_called_once()


# Блок 4: Детальні тести математики та логіки
class TestTSPSolverMath:

    def test_evaluate_logic_math(self):
        """Перевірка математики розрахунку довжини маршруту."""
        cities = [[0, 0], [0, 3], [4, 0]]
        solver = TSPSolver(cities=cities, cities_count=3, tournsize=2)
        route = [0, 1, 2]
        result_tuple = solver._TSPSolver__evaluate(route)
        assert result_tuple == (12.0,)

    def test_evaluate_error_too_short(self):
        """Перевірка помилки при занадто короткому маршруті."""
        solver = TSPSolver(cities_count=3, tournsize=2)
        short_route = [0]
        with pytest.raises(AttributeError, match="Route have to have at least two"):
            solver._TSPSolver__evaluate(short_route)

    def test_best_property_empty(self):
        """Що повертає властивість 'best', якщо алгоритм ще не запускався?"""
        solver = TSPSolver(cities_count=5, tournsize=2)
        assert solver.best == []

    @patch("random.random")
    @patch("random.randint")
    def test_indices_greedy_initialization(self, mock_randint, mock_random):
        """Тестування жадібної ініціалізації (__indices)."""
        cities = [[0, 0], [2, 0], [4, 0]]
        solver = TSPSolver(cities=cities, cities_count=3, p_qualitative=0.99, tournsize=2)
        mock_random.return_value = 0.5
        mock_randint.return_value = 0
        indices = solver._TSPSolver__indices()
        assert sorted(indices) == [0, 1, 2]

    def test_plot_progression_no_data(self):
        """Перевірка графіка прогресу, коли даних немає."""
        solver = TSPSolver(cities_count=5, tournsize=2)
        assert solver.plot_solving_progression() is None


# Покриття до 100%
class TestCoverageGapFill:

    @patch("solver_windows_service.tsp_solver.core.plt")
    def test_plot_best_route_with_data(self, mock_plt):
        """
        Тест методу plot_best_route. Симулюємо, що розв'язок знайдено, і викликаємо малювання.
        """
        # 3 міста
        cities = [[0, 0], [10, 0], [0, 10]]
        solver = TSPSolver(cities=cities, cities_count=3, tournsize=2)

        # Підсуваємо "знайдений" маршрут
        solver._TSPSolver__best_overall = [0, 1, 2]

        mock_plt.savefig.return_value = None
        result = solver.plot_best_route()

        # Перевірки
        assert isinstance(result, str)
        # Перевіряємо, що лінії малювалися (метод plot викликався)
        # Він має викликатися 3 рази (0->1, 1->2, 2->0)
        assert mock_plt.plot.call_count >= 3
        mock_plt.close.assert_called_once()

    def test_best_property_with_data(self):
        """
        Тест властивості 'best', коли дані є.
        """
        cities = [[1, 1], [2, 2]]
        solver = TSPSolver(cities=cities, cities_count=2, tournsize=2)
        solver._TSPSolver__best_overall = [0, 1]

        result = solver.best
        assert result == [[1, 1], [2, 2]]

    @patch("solver_windows_service.tsp_solver.core.plt")
    def test_plot_progression_avg_false(self, mock_plt):
        """
        Покриває гілку else у plot_solving_progression (коли avg=False).
        """
        solver = TSPSolver(cities_count=5, tournsize=2)
        solver._TSPSolver__best_overall = [0, 1]  # аби не спрацював перший return

        mock_logbook = MagicMock()
        mock_logbook.select.return_value = ([1], [1])  # min, avg
        solver._TSPSolver__logbook = mock_logbook

        mock_plt.savefig.return_value = None

        # Викликаємо з avg=False
        solver.plot_solving_progression(avg=False)

        # Перевіряємо, що plot викликався лише 1 раз (для min), а не 2
        assert mock_plt.plot.call_count == 1

    @patch("random.sample")
    @patch("random.random")
    def test_indices_simple_random(self, mock_random, mock_sample):
        """
        Покриває просту гілку в __indices:
        if random.random() < (1 - self.__p_qualitative): ...
        """
        solver = TSPSolver(cities_count=5, p_qualitative=0.5, tournsize=2)

        # Щоб умова (random < 0.5) була True, random має бути, наприклад, 0.1
        mock_random.return_value = 0.1

        # Налаштовуємо mock sample
        expected_indices = [4, 3, 2, 1, 0]
        mock_sample.return_value = expected_indices

        result = solver._TSPSolver__indices()

        assert result == expected_indices
        mock_sample.assert_called_once()

    @patch("random.randint")
    @patch("random.random")
    def test_indices_impossible_condition_error(self, mock_random, mock_randint):
        """
        Покриття raise ValueError("CITIES_COUNT is wrong") в __indices.
        """
        solver = TSPSolver(cities_count=3, p_qualitative=0.99, tournsize=2)
        # Йдемо в жадібну гілку
        mock_random.return_value = 0.9
        mock_randint.return_value = 0

        # Зробимо всі відстані ще БІЛЬШИМИ за min_dist
        big_val = 999999999999999
        solver._TSPSolver__distances = [
            [big_val, big_val, big_val],
            [big_val, big_val, big_val],
            [big_val, big_val, big_val]
        ]

        # Тепер цикл пройде, але жодне місто не задовольнить умову distance < min_dist
        # min_ind залишиться -1 -> raise ValueError
        with pytest.raises(ValueError, match="CITIES_COUNT is wrong"):
            solver._TSPSolver__indices()

"""
Unit tests for Chronos DAG Task Dependency Scheduler Engine.
"""

import time

import pytest

from chronos.scheduler.dag_engine import (
    CycleDetectedError,
    DAGScheduler,
    TaskExecutionError,
    TaskStatus,
)


class TestDAGSchedulerStructure:
    """Tests for DAG building, task registration, and edges."""

    def test_add_task(self):
        scheduler = DAGScheduler()
        task = scheduler.add_task("task1", lambda: 42)
        assert task.task_id == "task1"
        assert task.status == TaskStatus.PENDING
        assert "task1" in scheduler.tasks

    def test_duplicate_task_id_raises_error(self):
        scheduler = DAGScheduler()
        scheduler.add_task("task1", lambda: 1)
        with pytest.raises(ValueError, match="already exists"):
            scheduler.add_task("task1", lambda: 2)

    def test_add_dependency_valid(self):
        scheduler = DAGScheduler()
        scheduler.add_task("a", lambda: 1)
        scheduler.add_task("b", lambda: 2)
        scheduler.add_dependency("a", "b")

        order = scheduler.topological_sort()
        assert order.index("a") < order.index("b")

    def test_add_edge_alias(self):
        scheduler = DAGScheduler()
        scheduler.add_task("a", lambda: 1)
        scheduler.add_task("b", lambda: 2)
        scheduler.add_edge("a", "b")

        order = scheduler.topological_sort()
        assert order == ["a", "b"]

    def test_add_dependency_missing_node_raises_key_error(self):
        scheduler = DAGScheduler()
        scheduler.add_task("a", lambda: 1)
        with pytest.raises(KeyError):
            scheduler.add_dependency("a", "nonexistent")
        with pytest.raises(KeyError):
            scheduler.add_dependency("nonexistent", "a")

    def test_self_dependency_raises_cycle_error(self):
        scheduler = DAGScheduler()
        scheduler.add_task("a", lambda: 1)
        with pytest.raises(CycleDetectedError, match="Self-dependency"):
            scheduler.add_dependency("a", "a")


class TestCycleDetection:
    """Tests for cycle detection and topological sorting."""

    def test_no_cycle_linear(self):
        scheduler = DAGScheduler()
        scheduler.add_task("a", lambda: 1)
        scheduler.add_task("b", lambda: 2)
        scheduler.add_task("c", lambda: 3)
        scheduler.add_dependency("a", "b")
        scheduler.add_dependency("b", "c")

        assert not scheduler.has_cycle()
        assert scheduler.detect_cycle() is None
        assert scheduler.topological_sort() == ["a", "b", "c"]

    def test_simple_cycle_detected(self):
        scheduler = DAGScheduler()
        scheduler.add_task("a", lambda: 1)
        scheduler.add_task("b", lambda: 2)
        scheduler.add_dependency("a", "b")
        scheduler.add_dependency("b", "a")

        assert scheduler.has_cycle()
        cycle = scheduler.detect_cycle()
        assert cycle is not None
        assert len(cycle) >= 3

        with pytest.raises(CycleDetectedError):
            scheduler.topological_sort()

    def test_diamond_dag_no_cycle(self):
        scheduler = DAGScheduler()
        scheduler.add_task("start", lambda: 0)
        scheduler.add_task("branch1", lambda: 1)
        scheduler.add_task("branch2", lambda: 2)
        scheduler.add_task("join", lambda: 3)

        scheduler.add_dependency("start", "branch1")
        scheduler.add_dependency("start", "branch2")
        scheduler.add_dependency("branch1", "join")
        scheduler.add_dependency("branch2", "join")

        assert not scheduler.has_cycle()
        order = scheduler.topological_sort()
        assert order.index("start") == 0
        assert order.index("join") == 3

    def test_complex_cycle_detected(self):
        scheduler = DAGScheduler()
        for i in range(5):
            scheduler.add_task(f"t{i}", lambda: i)

        scheduler.add_dependency("t0", "t1")
        scheduler.add_dependency("t1", "t2")
        scheduler.add_dependency("t2", "t3")
        scheduler.add_dependency("t3", "t1")  # Cycle t1 -> t2 -> t3 -> t1

        assert scheduler.has_cycle()
        with pytest.raises(CycleDetectedError):
            scheduler.topological_sort()


class TestDAGExecution:
    """Tests for parallel and topological execution."""

    def test_empty_dag(self):
        scheduler = DAGScheduler()
        results = scheduler.execute()
        assert results == {}

    def test_single_task_execution(self):
        scheduler = DAGScheduler()
        scheduler.add_task("only", lambda: 100)
        results = scheduler.execute()
        assert results == {"only": 100}
        assert scheduler.get_task("only").status == TaskStatus.COMPLETED

    def test_linear_dependency_execution(self):
        scheduler = DAGScheduler()
        order = []

        def make_fn(name, val):
            def fn():
                order.append(name)
                return val
            return fn

        scheduler.add_task("a", make_fn("a", 1))
        scheduler.add_task("b", make_fn("b", 2))
        scheduler.add_task("c", make_fn("c", 3))

        scheduler.add_dependency("a", "b")
        scheduler.add_dependency("b", "c")

        results = scheduler.execute()
        assert results == {"a": 1, "b": 2, "c": 3}
        assert order == ["a", "b", "c"]

    def test_parallel_execution(self):
        scheduler = DAGScheduler()
        start_time = time.time()

        def slow_task():
            time.sleep(0.05)
            return True

        # 4 independent tasks
        for i in range(4):
            scheduler.add_task(f"task_{i}", slow_task)

        results = scheduler.execute(max_workers=4)
        elapsed = time.time() - start_time

        assert len(results) == 4
        # In parallel with 4 workers, 4x50ms should take < 150ms
        assert elapsed < 0.18

    def test_dependency_results_passed(self):
        scheduler = DAGScheduler()

        scheduler.add_task("step1", lambda: 10)
        scheduler.add_task("step2", lambda: 20)

        def aggregator(dependency_results):
            return dependency_results["step1"] + dependency_results["step2"]

        scheduler.add_task("sum", aggregator)
        scheduler.add_dependency("step1", "sum")
        scheduler.add_dependency("step2", "sum")

        results = scheduler.execute()
        assert results["sum"] == 30

    def test_task_failure_and_cancellation(self):
        scheduler = DAGScheduler()

        def fail_fn():
            raise RuntimeError("Task crashed")

        scheduler.add_task("root", lambda: "ok")
        scheduler.add_task("faulty", fail_fn)
        scheduler.add_task("dependent", lambda: "never reached")

        scheduler.add_dependency("root", "faulty")
        scheduler.add_dependency("faulty", "dependent")

        with pytest.raises(TaskExecutionError) as exc_info:
            scheduler.execute(fail_fast=True)

        assert exc_info.value.task_id == "faulty"
        assert scheduler.get_task("faulty").status == TaskStatus.FAILED
        assert scheduler.get_task("dependent").status == TaskStatus.CANCELLED

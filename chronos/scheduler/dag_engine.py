"""
Distributed Task Dependency DAG Execution Engine for Chronos.

Provides DAG-based task dependency modeling, Kahn's algorithm / DFS cycle detection,
topological sorting, and parallel concurrent task execution.
"""

from __future__ import annotations

import concurrent.futures
import inspect
import threading
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set


class TaskStatus(Enum):
    """Execution status of a DAG task."""

    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()


class CycleDetectedError(Exception):
    """Raised when a dependency cycle is detected in the DAG."""

    def __init__(self, message: str, cycle: Optional[List[str]] = None) -> None:
        super().__init__(message)
        self.cycle = cycle or []


class TaskExecutionError(Exception):
    """Raised when one or more DAG tasks fail during execution."""

    def __init__(self, message: str, task_id: str, original_error: Exception) -> None:
        super().__init__(message)
        self.task_id = task_id
        self.original_error = original_error


@dataclass
class Task:
    """Represents an executable node in the task dependency DAG."""

    task_id: str
    fn: Callable[..., Any]
    args: tuple = ()
    kwargs: dict = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: Optional[Exception] = None

    def execute(self, dependency_results: Optional[Dict[str, Any]] = None) -> Any:
        """
        Execute the task callable.

        If the task accepts a 'dependency_results' parameter, it is provided.
        """
        self.status = TaskStatus.RUNNING
        try:
            call_kwargs = dict(self.kwargs)
            sig = inspect.signature(self.fn)
            if "dependency_results" in sig.parameters and dependency_results is not None:
                call_kwargs["dependency_results"] = dependency_results

            self.result = self.fn(*self.args, **call_kwargs)
            self.status = TaskStatus.COMPLETED
            return self.result
        except Exception as e:
            self.status = TaskStatus.FAILED
            self.error = e
            raise


class DAGScheduler:
    """
    Direct Acyclic Graph (DAG) Task Scheduler.

    Supports adding task nodes, dependency edges, cycle detection using
    Kahn's algorithm and DFS, topological sorting, and parallel execution
    of independent tasks.
    """

    def __init__(self) -> None:
        self._tasks: Dict[str, Task] = {}
        # adjacency list: upstream -> set of downstream tasks
        self._downstream: Dict[str, Set[str]] = {}
        # reverse adjacency list: downstream -> set of upstream dependencies
        self._upstream: Dict[str, Set[str]] = {}
        self._lock = threading.RLock()

    def add_task(
        self,
        task_id: str,
        fn: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Task:
        """
        Add a task node to the DAG.

        Args:
            task_id: Unique identifier for the task.
            fn: Callable to be executed.
            *args: Positional arguments for fn.
            **kwargs: Keyword arguments for fn.

        Returns:
            The created Task instance.
        """
        with self._lock:
            if task_id in self._tasks:
                raise ValueError(f"Task with ID '{task_id}' already exists in DAG")

            task = Task(task_id=task_id, fn=fn, args=args, kwargs=kwargs)
            self._tasks[task_id] = task
            self._downstream[task_id] = set()
            self._upstream[task_id] = set()
            return task

    def add_dependency(self, upstream_id: str, downstream_id: str) -> None:
        """
        Add a directed dependency edge from upstream_id to downstream_id.
        (downstream_id depends on upstream_id).

        Args:
            upstream_id: ID of the prerequisite task.
            downstream_id: ID of the dependent task.
        """
        with self._lock:
            if upstream_id not in self._tasks:
                raise KeyError(f"Upstream task '{upstream_id}' does not exist in DAG")
            if downstream_id not in self._tasks:
                raise KeyError(f"Downstream task '{downstream_id}' does not exist in DAG")
            if upstream_id == downstream_id:
                raise CycleDetectedError(
                    f"Self-dependency detected on task '{upstream_id}'",
                    cycle=[upstream_id, upstream_id],
                )

            self._downstream[upstream_id].add(downstream_id)
            self._upstream[downstream_id].add(upstream_id)

    def add_edge(self, upstream_id: str, downstream_id: str) -> None:
        """Alias for add_dependency."""
        self.add_dependency(upstream_id, downstream_id)

    def get_task(self, task_id: str) -> Task:
        """Retrieve task by its ID."""
        with self._lock:
            if task_id not in self._tasks:
                raise KeyError(f"Task '{task_id}' not found")
            return self._tasks[task_id]

    @property
    def tasks(self) -> Dict[str, Task]:
        """Return shallow copy of registered tasks."""
        with self._lock:
            return dict(self._tasks)

    def detect_cycle(self) -> Optional[List[str]]:
        """
        Detect whether a cycle exists using Depth-First Search (DFS).

        Returns:
            A list representing the cycle path if found, or None.
        """
        with self._lock:
            visited: Dict[str, int] = {}  # 0 = unvisited, 1 = visiting (in stack), 2 = visited
            parent: Dict[str, Optional[str]] = {}
            cycle_path: Optional[List[str]] = None

            def dfs(node: str, stack: List[str]) -> bool:
                nonlocal cycle_path
                visited[node] = 1
                stack.append(node)

                for neighbor in sorted(self._downstream.get(node, set())):
                    if visited.get(neighbor, 0) == 1:
                        # Cycle found
                        idx = stack.index(neighbor)
                        cycle_path = stack[idx:] + [neighbor]
                        return True
                    if visited.get(neighbor, 0) == 0:
                        parent[neighbor] = node
                        if dfs(neighbor, stack):
                            return True

                stack.pop()
                visited[node] = 2
                return False

            for task_id in sorted(self._tasks.keys()):
                if visited.get(task_id, 0) == 0:
                    if dfs(task_id, []):
                        return cycle_path

            return None

    def has_cycle(self) -> bool:
        """Return True if a cycle is detected, False otherwise."""
        return self.detect_cycle() is not None

    def topological_sort(self) -> List[str]:
        """
        Compute a valid topological ordering of tasks using Kahn's algorithm.

        Returns:
            List of task IDs in topological execution order.

        Raises:
            CycleDetectedError: If a cycle exists in the DAG.
        """
        with self._lock:
            cycle = self.detect_cycle()
            if cycle:
                raise CycleDetectedError(
                    f"Cycle detected in DAG: {' -> '.join(cycle)}",
                    cycle=cycle,
                )

            # In-degrees for all tasks
            in_degree = {task_id: len(self._upstream[task_id]) for task_id in self._tasks}
            queue = [task_id for task_id, deg in in_degree.items() if deg == 0]
            queue.sort()

            result: List[str] = []

            while queue:
                curr = queue.pop(0)
                result.append(curr)

                for neighbor in sorted(self._downstream[curr]):
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)
                        queue.sort()

            if len(result) != len(self._tasks):
                raise CycleDetectedError("Cycle detected during topological sorting")

            return result

    def reset(self) -> None:
        """Reset execution status and results for all tasks."""
        with self._lock:
            for task in self._tasks.values():
                task.status = TaskStatus.PENDING
                task.result = None
                task.error = None

    def execute(
        self,
        max_workers: Optional[int] = None,
        fail_fast: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute tasks in topological order while parallelizing independent nodes.

        Uses dynamic ready-queue scheduling where tasks become eligible for execution
        the moment all their upstream dependencies complete successfully.

        Args:
            max_workers: Maximum number of worker threads in the thread pool.
            fail_fast: If True, halts pending tasks when a task fails and raises TaskExecutionError.

        Returns:
            Dictionary mapping task_id to execution result.

        Raises:
            CycleDetectedError: If the DAG contains a cycle.
            TaskExecutionError: If any task raises an unhandled exception.
        """
        with self._lock:
            # Detect cycles prior to execution
            self.topological_sort()
            self.reset()

            if not self._tasks:
                return {}

            results: Dict[str, Any] = {}
            errors: Dict[str, Exception] = {}
            completed_tasks: Set[str] = set()
            in_degrees = {tid: len(self._upstream[tid]) for tid in self._tasks}

            state_lock = threading.Lock()
            cv = threading.Condition(state_lock)

            ready_tasks: List[str] = [tid for tid, deg in in_degrees.items() if deg == 0]
            aborted = False

            def worker_fn(task_id: str) -> None:
                nonlocal aborted
                task = self._tasks[task_id]

                # Fetch results of immediate upstream dependencies
                with state_lock:
                    dep_results = {
                        up_id: results[up_id]
                        for up_id in self._upstream[task_id]
                        if up_id in results
                    }

                try:
                    res = task.execute(dependency_results=dep_results)
                    with state_lock:
                        results[task_id] = res
                        completed_tasks.add(task_id)

                        # Enqueue newly ready downstream dependents
                        for down_id in self._downstream[task_id]:
                            in_degrees[down_id] -= 1
                            if in_degrees[down_id] == 0 and not aborted:
                                ready_tasks.append(down_id)

                        cv.notify_all()
                except Exception as ex:
                    with state_lock:
                        errors[task_id] = ex
                        if fail_fast:
                            aborted = True
                            # Cancel unstarted downstream tasks
                            self._cancel_downstream(task_id)
                        cv.notify_all()

            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                while True:
                    with state_lock:
                        if aborted and fail_fast:
                            break

                        if len(completed_tasks) + len(errors) == len(self._tasks):
                            break

                        tasks_to_launch = list(ready_tasks)
                        ready_tasks.clear()

                    if tasks_to_launch:
                        for tid in tasks_to_launch:
                            executor.submit(worker_fn, tid)

                    with state_lock:
                        if (
                            not ready_tasks
                            and len(completed_tasks) + len(errors) < len(self._tasks)
                            and not (aborted and fail_fast)
                        ):
                            cv.wait(timeout=0.1)

            if errors:
                failed_id, first_err = next(iter(errors.items()))
                raise TaskExecutionError(
                    f"DAG execution failed on task '{failed_id}': {first_err}",
                    task_id=failed_id,
                    original_error=first_err,
                )

            return results

    def _cancel_downstream(self, failed_task_id: str) -> None:
        """Mark all downstream transitive dependents of failed_task_id as CANCELLED."""
        to_visit = list(self._downstream.get(failed_task_id, set()))
        visited: Set[str] = set()

        while to_visit:
            curr = to_visit.pop(0)
            if curr not in visited:
                visited.add(curr)
                task = self._tasks[curr]
                if task.status == TaskStatus.PENDING:
                    task.status = TaskStatus.CANCELLED
                to_visit.extend(self._downstream.get(curr, set()))

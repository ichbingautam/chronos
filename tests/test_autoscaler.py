"""
Unit tests for Chronos Elastic Worker Auto-Scaling Manager.
"""

import time

import pytest

from chronos.distributed.autoscaler import (
    AutoscalerConfig,
    ElasticWorkerPoolManager,
    WorkerNode,
    WorkerNodeState,
)


class TestAutoscalerInitialization:
    def test_default_initialization_warms_up_min_workers(self):
        config = AutoscalerConfig(min_workers=2, max_workers=5)
        manager = ElasticWorkerPoolManager(config)

        assert manager.worker_count == 2
        assert manager.idle_worker_count == 2
        workers = manager.get_all_workers()
        assert len(workers) == 2
        for w in workers:
            assert w.state == WorkerNodeState.IDLE
            assert w.idle_since is not None

    def test_config_validation(self):
        with pytest.raises(ValueError, match="min_workers"):
            AutoscalerConfig(min_workers=-1)

        with pytest.raises(ValueError, match="max_workers"):
            AutoscalerConfig(min_workers=5, max_workers=3)

        with pytest.raises(ValueError, match="worker_capacity"):
            AutoscalerConfig(worker_capacity=0)

        with pytest.raises(ValueError, match="target_latency_sec"):
            AutoscalerConfig(target_latency_sec=0)


class TestAutoScalingUp:
    def test_scale_up_on_ready_task_backlog(self):
        config = AutoscalerConfig(
            min_workers=1,
            max_workers=6,
            scale_up_threshold_tasks=5,
            scale_up_step=2,
            target_latency_sec=5.0,
        )
        manager = ElasticWorkerPoolManager(config)
        assert manager.worker_count == 1

        # Queue depth below threshold: no scale up
        action = manager.evaluate_and_scale(ready_task_count=3, average_queue_latency_sec=1.0)
        assert action["scaled_up"] == 0
        assert manager.worker_count == 1

        # Queue depth reaches threshold: scale up by step (2)
        action = manager.evaluate_and_scale(ready_task_count=6, average_queue_latency_sec=1.0)
        assert action["scaled_up"] == 2
        assert manager.worker_count == 3
        assert len(action["provisioned_workers"]) == 2

    def test_scale_up_on_latency_sla_breach(self):
        config = AutoscalerConfig(
            min_workers=1,
            max_workers=4,
            scale_up_threshold_tasks=10,
            target_latency_sec=2.0,
            scale_up_step=1,
        )
        manager = ElasticWorkerPoolManager(config)
        assert manager.worker_count == 1

        # Latency breaches SLA target (3.5s > 2.0s)
        action = manager.evaluate_and_scale(ready_task_count=2, average_queue_latency_sec=3.5)
        assert action["scaled_up"] == 1
        assert manager.worker_count == 2

    def test_scale_up_obeys_max_workers_ceiling(self):
        config = AutoscalerConfig(
            min_workers=2,
            max_workers=3,
            scale_up_threshold_tasks=2,
            scale_up_step=5,  # request more than max headroom
        )
        manager = ElasticWorkerPoolManager(config)
        assert manager.worker_count == 2

        action = manager.evaluate_and_scale(ready_task_count=10, average_queue_latency_sec=1.0)
        # Only 1 worker can be added to reach max 3
        assert action["scaled_up"] == 1
        assert manager.worker_count == 3

        # Next evaluate: ceiling reached
        action2 = manager.evaluate_and_scale(ready_task_count=20, average_queue_latency_sec=10.0)
        assert action2["scaled_up"] == 0
        assert manager.worker_count == 3


class TestAutoScalingDown:
    def test_scale_down_after_cooldown(self):
        config = AutoscalerConfig(
            min_workers=1,
            max_workers=4,
            scale_down_cooldown_sec=0.05,  # 50ms cooldown for fast test
            scale_down_step=1,
        )
        manager = ElasticWorkerPoolManager(config)
        # Register a second worker
        w2 = manager.register_worker("worker-2")
        assert manager.worker_count == 2

        # Immediate evaluate without waiting for cooldown -> no scale down
        action = manager.evaluate_and_scale(ready_task_count=0, average_queue_latency_sec=0.0)
        assert action["scaled_down"] == 0
        assert manager.worker_count == 2

        # Sleep to exceed cooldown
        time.sleep(0.06)

        action = manager.evaluate_and_scale(ready_task_count=0, average_queue_latency_sec=0.0)
        assert action["scaled_down"] == 1
        assert manager.worker_count == 1

    def test_scale_down_obeys_min_workers_floor(self):
        config = AutoscalerConfig(
            min_workers=2,
            max_workers=5,
            scale_down_cooldown_sec=0.01,
        )
        manager = ElasticWorkerPoolManager(config)
        assert manager.worker_count == 2

        time.sleep(0.02)
        # Queue is empty, workers idle, but at min_workers -> no scale down
        action = manager.evaluate_and_scale(ready_task_count=0, average_queue_latency_sec=0.0)
        assert action["scaled_down"] == 0
        assert manager.worker_count == 2

    def test_graceful_scale_down_with_inflight_tasks(self):
        config = AutoscalerConfig(min_workers=1, max_workers=3)
        manager = ElasticWorkerPoolManager(config)
        w2 = manager.register_worker("worker-busy")
        assert manager.worker_count == 2

        # Assign task to worker-busy
        assigned_id = manager.acquire_worker_for_task("task-100")
        assert assigned_id is not None
        node = manager.get_worker(assigned_id)
        assert node.in_flight_count == 1

        # Attempt to unregister worker with in-flight task non-forcefully
        unregistered = manager.unregister_worker(assigned_id, force=False)
        assert not unregistered  # should not immediately delete
        assert node.state == WorkerNodeState.TERMINATING

        # Worker cannot take new tasks while terminating
        assert not node.is_available

        # Complete task: node drains and is terminated cleanly
        manager.release_task(assigned_id, "task-100")
        assert manager.get_worker(assigned_id) is None


class TestTaskAllocationAndMetrics:
    def test_task_capacity_allocation(self):
        config = AutoscalerConfig(min_workers=1, max_workers=2, worker_capacity=2)
        manager = ElasticWorkerPoolManager(config)

        # Allocate 2 tasks to single worker
        w1_task1 = manager.acquire_worker_for_task("t1")
        w1_task2 = manager.acquire_worker_for_task("t2")
        assert w1_task1 is not None
        assert w1_task1 == w1_task2

        # Capacity full (2/2): next acquisition returns None
        w1_task3 = manager.acquire_worker_for_task("t3")
        assert w1_task3 is None

        # Release one task -> can acquire again
        manager.release_task(w1_task1, "t1")
        w1_task4 = manager.acquire_worker_for_task("t4")
        assert w1_task4 == w1_task1

    def test_metrics_reporting(self):
        config = AutoscalerConfig(min_workers=2, max_workers=5, worker_capacity=2)
        manager = ElasticWorkerPoolManager(config)

        # 0 tasks
        m0 = manager.get_metrics(ready_task_count=3, average_queue_latency_sec=1.2)
        assert m0.ready_task_count == 3
        assert m0.in_flight_task_count == 0
        assert m0.active_worker_count == 2
        assert m0.idle_worker_count == 2
        assert m0.worker_utilization == 0.0

        # Assign 1 task
        wid = manager.acquire_worker_for_task("task-1")
        m1 = manager.get_metrics(ready_task_count=2, average_queue_latency_sec=0.8)
        assert m1.in_flight_task_count == 1
        assert m1.idle_worker_count == 1  # 1 busy, 1 idle
        assert m1.worker_utilization == 0.25  # 1 task / (2 workers * 2 capacity)

    def test_custom_provision_and_terminate_hooks(self):
        provisioned = []
        terminated = []

        config = AutoscalerConfig(min_workers=1, max_workers=3)
        manager = ElasticWorkerPoolManager(
            config=config,
            provision_fn=lambda wid: provisioned.append(wid),
            terminate_fn=lambda wid: terminated.append(wid),
        )

        assert len(provisioned) == 1  # initial min_worker

        new_worker = manager.register_worker("hook-worker")
        assert "hook-worker" in provisioned

        manager.unregister_worker("hook-worker", force=True)
        assert "hook-worker" in terminated

"""
Message protocols for distributed Chronos.

Defines the message types and serialization for coordinator-worker communication.
Uses pickle for PyTorch tensor compatibility.
"""

import pickle
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, Optional

import torch
from torch import Tensor


class MessageType(Enum):
    """Types of messages in the Chronos protocol."""

    # Worker -> Coordinator
    CHECKOUT_REQUEST = auto()      # Request outer params
    COMMIT_REQUEST = auto()        # Submit trajectory
    HEARTBEAT = auto()             # Keep-alive
    WORKER_REGISTER = auto()       # Register new worker
    WORKER_DEREGISTER = auto()     # Unregister worker

    # Coordinator -> Worker
    CHECKOUT_RESPONSE = auto()     # Outer params + version
    COMMIT_RESPONSE = auto()       # Accept/reject + weight
    ERROR_RESPONSE = auto()        # Error message

    # Broadcast (Coordinator -> All Workers)
    PARAMS_UPDATED = auto()        # New outer params available
    SHUTDOWN = auto()              # Shutdown command


@dataclass
class CheckoutRequest:
    """Request to checkout current outer parameters."""

    worker_id: str
    blocking: bool = True

    def serialize(self) -> bytes:
        return pickle.dumps({
            "type": MessageType.CHECKOUT_REQUEST,
            "worker_id": self.worker_id,
            "blocking": self.blocking,
        })

    @classmethod
    def deserialize(cls, data: bytes) -> "CheckoutRequest":
        obj = pickle.loads(data)
        return cls(
            worker_id=obj["worker_id"],
            blocking=obj.get("blocking", True)
        )


@dataclass
class CheckoutResponse:
    """Response containing outer parameters and version."""

    success: bool
    version: Optional[int] = None
    outer_params: Optional[Dict[str, Tensor]] = None
    error: Optional[str] = None

    def serialize(self) -> bytes:
        return pickle.dumps({
            "type": MessageType.CHECKOUT_RESPONSE,
            "success": self.success,
            "version": self.version,
            "outer_params": self.outer_params,
            "error": self.error,
        })

    @classmethod
    def deserialize(cls, data: bytes) -> "CheckoutResponse":
        obj = pickle.loads(data)
        return cls(
            success=obj["success"],
            version=obj.get("version"),
            outer_params=obj.get("outer_params"),
            error=obj.get("error")
        )


@dataclass
class CommitRequest:
    """Request to commit a completed trajectory."""

    worker_id: str
    version: int
    trajectory_data: Dict[str, Any]  # Serializable trajectory info

    def serialize(self) -> bytes:
        return pickle.dumps({
            "type": MessageType.COMMIT_REQUEST,
            "worker_id": self.worker_id,
            "version": self.version,
            "trajectory_data": self.trajectory_data,
        })

    @classmethod
    def deserialize(cls, data: bytes) -> "CommitRequest":
        obj = pickle.loads(data)
        return cls(
            worker_id=obj["worker_id"],
            version=obj["version"],
            trajectory_data=obj["trajectory_data"]
        )


@dataclass
class CommitResponse:
    """Response to trajectory commit."""

    accepted: bool
    staleness_weight: float = 1.0
    new_version: Optional[int] = None
    error: Optional[str] = None

    def serialize(self) -> bytes:
        return pickle.dumps({
            "type": MessageType.COMMIT_RESPONSE,
            "accepted": self.accepted,
            "staleness_weight": self.staleness_weight,
            "new_version": self.new_version,
            "error": self.error,
        })

    @classmethod
    def deserialize(cls, data: bytes) -> "CommitResponse":
        obj = pickle.loads(data)
        return cls(
            accepted=obj["accepted"],
            staleness_weight=obj.get("staleness_weight", 1.0),
            new_version=obj.get("new_version"),
            error=obj.get("error")
        )


@dataclass
class HeartbeatRequest:
    """Heartbeat to keep worker registered."""

    worker_id: str
    current_version: Optional[int] = None
    status: str = "idle"  # idle, computing, waiting

    def serialize(self) -> bytes:
        return pickle.dumps({
            "type": MessageType.HEARTBEAT,
            "worker_id": self.worker_id,
            "current_version": self.current_version,
            "status": self.status,
        })


@dataclass
class ErrorResponse:
    """Error response from coordinator."""

    error: str
    code: int = 500

    def serialize(self) -> bytes:
        return pickle.dumps({
            "type": MessageType.ERROR_RESPONSE,
            "error": self.error,
            "code": self.code,
        })


def parse_message(data: bytes) -> tuple:
    """
    Parse incoming message and return (MessageType, payload).

    Returns:
        Tuple of (MessageType, deserialized payload dict)
    """
    obj = pickle.loads(data)
    msg_type = obj.get("type")
    return msg_type, obj

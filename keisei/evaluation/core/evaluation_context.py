"""
Evaluation context and metadata structures.

This module defines the context information that accompanies an evaluation session,
including agent information, environment details, and session metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from .evaluation_config import EvaluationConfig


@dataclass
class AgentInfo:
    """Information about the agent being evaluated."""

    name: str
    checkpoint_path: Optional[str] = None
    model_type: Optional[str] = None
    training_timesteps: Optional[int] = None
    version: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert AgentInfo to dictionary for serialization."""
        return {
            "name": self.name,
            "checkpoint_path": self.checkpoint_path,
            "model_type": self.model_type,
            "training_timesteps": self.training_timesteps,
            "version": self.version,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentInfo":
        """Create AgentInfo from dictionary."""
        return cls(
            name=data.get("name", "UnknownAgent"),
            checkpoint_path=data.get("checkpoint_path"),
            model_type=data.get("model_type"),
            training_timesteps=data.get("training_timesteps"),
            version=data.get("version"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class OpponentInfo:
    """Information about an opponent in evaluation."""

    name: str
    type: str  # "random", "heuristic", "ppo", etc.
    checkpoint_path: Optional[str] = None
    difficulty_level: Optional[float] = None
    version: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert OpponentInfo to dictionary for serialization."""
        return {
            "name": self.name,
            "type": self.type,
            "checkpoint_path": self.checkpoint_path,
            "difficulty_level": self.difficulty_level,
            "version": self.version,
            "metadata": self.metadata.copy(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OpponentInfo":
        """Create OpponentInfo from dictionary."""
        return cls(
            name=data.get("name", "UnknownOpponent"),
            type=data.get("type", "unknown"),
            checkpoint_path=data.get("checkpoint_path"),
            difficulty_level=data.get("difficulty_level"),
            version=data.get("version"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class EvaluationContext:
    """Context information for an evaluation session."""

    session_id: str
    timestamp: datetime
    agent_info: AgentInfo
    configuration: EvaluationConfig
    environment_info: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "timestamp": self.timestamp.isoformat(),
            "agent_info": self.agent_info.to_dict(),  # Use AgentInfo.to_dict()
            "configuration": self.configuration.to_dict(),
            "environment_info": self.environment_info,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(
        cls, data: Dict[str, Any], config: "EvaluationConfig"
    ) -> EvaluationContext:
        """Create context from dictionary."""
        agent_data = data["agent_info"]
        agent_info = AgentInfo.from_dict(agent_data)  # Use AgentInfo.from_dict()

        return cls(
            session_id=data["session_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            agent_info=agent_info,
            configuration=config,
            environment_info=data.get("environment_info", {}),
            metadata=data.get("metadata", {}),
        )

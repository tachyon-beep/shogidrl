"""
compatibility_mixin.py: Provides backward compatibility properties and methods for the Trainer class.
"""

import os
from typing import Any, Callable, Dict, List, Optional


class CompatibilityMixin:
    """
    Mixin class providing backward compatibility properties and methods.
    Keeps the main Trainer class clean while preserving API compatibility.
    """

    # === Model Properties (delegated to ModelManager) ===
    @property
    def feature_spec(self):
        """Access the feature spec through ModelManager."""
        model_manager = getattr(self, "model_manager", None)
        return (
            getattr(model_manager, "feature_spec", None)
            if model_manager
            else None
        )

    @property
    def obs_shape(self):
        """Access the observation shape through ModelManager."""
        model_manager = getattr(self, "model_manager", None)
        return (
            getattr(model_manager, "obs_shape", None)
            if model_manager
            else None
        )

    @property
    def tower_depth(self):
        """Access the tower depth through ModelManager."""
        model_manager = getattr(self, "model_manager", None)
        return (
            getattr(model_manager, "tower_depth", None)
            if model_manager
            else None
        )

    @property
    def tower_width(self):
        """Access the tower width through ModelManager."""
        model_manager = getattr(self, "model_manager", None)
        return (
            getattr(model_manager, "tower_width", None)
            if model_manager
            else None
        )

    @property
    def se_ratio(self):
        """Access the SE ratio through ModelManager."""
        model_manager = getattr(self, "model_manager", None)
        return (
            getattr(model_manager, "se_ratio", None)
            if model_manager
            else None
        )

    # === Metrics Properties (delegated to MetricsManager) ===
    @property
    def global_timestep(self) -> int:
        """Current global timestep (backward compatibility)."""
        metrics_manager = getattr(self, "metrics_manager", None)
        return getattr(metrics_manager, "global_timestep", 0) if metrics_manager else 0
    
    @global_timestep.setter
    def global_timestep(self, value: int) -> None:
        """Set global timestep (backward compatibility)."""
        metrics_manager = getattr(self, "metrics_manager", None)
        if metrics_manager:
            metrics_manager.global_timestep = value
    
    @property
    def total_episodes_completed(self) -> int:
        """Total episodes completed (backward compatibility)."""
        metrics_manager = getattr(self, "metrics_manager", None)
        return getattr(metrics_manager, "total_episodes_completed", 0) if metrics_manager else 0
    
    @total_episodes_completed.setter
    def total_episodes_completed(self, value: int) -> None:
        """Set total episodes completed (backward compatibility)."""
        metrics_manager = getattr(self, "metrics_manager", None)
        if metrics_manager:
            metrics_manager.total_episodes_completed = value
    
    @property
    def black_wins(self) -> int:
        """Number of black wins (backward compatibility)."""
        metrics_manager = getattr(self, "metrics_manager", None)
        return getattr(metrics_manager, "black_wins", 0) if metrics_manager else 0
    
    @black_wins.setter
    def black_wins(self, value: int) -> None:
        """Set black wins (backward compatibility)."""
        metrics_manager = getattr(self, "metrics_manager", None)
        if metrics_manager:
            metrics_manager.black_wins = value
    
    @property
    def white_wins(self) -> int:
        """Number of white wins (backward compatibility)."""
        metrics_manager = getattr(self, "metrics_manager", None)
        return getattr(metrics_manager, "white_wins", 0) if metrics_manager else 0
    
    @white_wins.setter
    def white_wins(self, value: int) -> None:
        """Set white wins (backward compatibility)."""
        metrics_manager = getattr(self, "metrics_manager", None)
        if metrics_manager:
            metrics_manager.white_wins = value
    
    @property
    def draws(self) -> int:
        """Number of draws (backward compatibility)."""
        metrics_manager = getattr(self, "metrics_manager", None)
        return getattr(metrics_manager, "draws", 0) if metrics_manager else 0
    
    @draws.setter
    def draws(self, value: int) -> None:
        """Set draws (backward compatibility)."""
        metrics_manager = getattr(self, "metrics_manager", None)
        if metrics_manager:
            metrics_manager.draws = value

    # === Model Artifact Creation (delegated to ModelManager) ===
    def _create_model_artifact(
        self,
        model_path: str,
        artifact_name: Optional[str] = None,
        artifact_type: str = "model",
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        aliases: Optional[List[str]] = None,
        log_both: Optional[Callable] = None,
    ) -> bool:
        """Backward compatibility method - delegates to ModelManager."""
        # Use default artifact name if not provided
        if artifact_name is None:
            artifact_name = os.path.basename(model_path)

        # Use default description if not provided
        if description is None:
            # Use trainer's run_name for backward compatibility (tests set this manually)
            session_manager = getattr(self, "session_manager", None)
            session_run_name = getattr(session_manager, "run_name", "unknown") if session_manager else "unknown"
            run_name = getattr(self, "run_name", session_run_name)
            description = f"Model checkpoint from run {run_name}"

        # Use trainer's run_name for artifact naming (for backward compatibility with tests)
        session_manager = getattr(self, "session_manager", None)
        session_run_name = getattr(session_manager, "run_name", "unknown") if session_manager else "unknown"
        run_name = getattr(self, "run_name", session_run_name)

        # Store original logger function and temporarily replace if log_both provided
        model_manager = getattr(self, "model_manager", None)
        if not model_manager:
            return False
            
        original_logger = getattr(model_manager, "logger_func", None)
        if log_both:
            # Create a wrapper that detects error messages and adds log_level="error"
            def logger_wrapper(message):
                if "Error creating W&B artifact" in message:
                    return log_both(message, log_level="error")
                else:
                    return log_both(message)

            model_manager.logger_func = logger_wrapper

        try:
            result = model_manager.create_model_artifact(
                model_path=model_path,
                artifact_name=artifact_name,
                run_name=run_name,
                is_wandb_active=getattr(session_manager, "is_wandb_active", False) if session_manager else False,
                artifact_type=artifact_type,
                description=description,
                metadata=metadata,
                aliases=aliases,
            )
        finally:
            # Restore original logger function
            if original_logger is not None:
                model_manager.logger_func = original_logger

        return result

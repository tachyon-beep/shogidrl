"""
DEPRECATED: This module has been replaced by unified configuration in config_schema.py

The evaluation system now uses the main system's Pydantic-based EvaluationConfig
to ensure proper integration with the training pipeline.

All imports from this module should be updated to use:
    from keisei.config_schema import EvaluationConfig

This file is maintained temporarily for migration purposes and will be removed
in a future version.
"""

import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Import for type checking only to avoid circular imports
    from keisei.config_schema import EvaluationConfig as UnifiedEvaluationConfig

# Re-export the unified config for backward compatibility during migration
try:
    from keisei.config_schema import EvaluationConfig
    
    # Issue deprecation warning
    warnings.warn(
        "keisei.evaluation.core.evaluation_config is deprecated. "
        "Use keisei.config_schema.EvaluationConfig instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Maintain backward compatibility for a short migration period
    __all__ = ["EvaluationConfig"]
    
except ImportError as e:
    raise ImportError(
        "Failed to import unified EvaluationConfig from config_schema. "
        "The evaluation system configuration has been unified. "
        "Please update imports to use: from keisei.config_schema import EvaluationConfig"
    ) from e
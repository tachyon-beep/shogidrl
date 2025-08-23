"""
Performance tests for torch.compile integration.

Tests the torch.compile optimization framework including:
- Model compilation validation
- Performance benchmarking
- Numerical equivalence verification
- Configuration handling
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock

from keisei.config_schema import TrainingConfig
from keisei.utils.performance_benchmarker import PerformanceBenchmarker, create_benchmarker
from keisei.utils.compilation_validator import CompilationValidator, safe_compile_model
from keisei.training.models.resnet_tower import ActorCriticResTower


class MockActorCritic(nn.Module):
    """Mock ActorCritic model for testing."""
    
    def __init__(self, input_channels=46, num_actions=13527):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
        self.policy_head = nn.Linear(16 * 9 * 9, num_actions)
        self.value_head = nn.Linear(16 * 9 * 9, 1)
    
    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = x.flatten(start_dim=1)
        policy = self.policy_head(x)
        value = self.value_head(x).squeeze(-1)
        return policy, value
    
    def get_action_and_value(self, obs, action=None, deterministic=False):
        """Mock implementation for protocol compliance."""
        policy_logits, value = self.forward(obs)
        if action is None:
            # Sample action (mock implementation)
            action = torch.multinomial(torch.softmax(policy_logits, dim=-1), 1).squeeze(-1)
        log_prob = torch.log_softmax(policy_logits, dim=-1).gather(-1, action.unsqueeze(-1)).squeeze(-1)
        return action, log_prob, value
    
    def evaluate_actions(self, obs, actions, values=None):
        """Mock implementation for protocol compliance."""
        policy_logits, value = self.forward(obs)
        log_probs = torch.log_softmax(policy_logits, dim=-1).gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        entropy = -(torch.softmax(policy_logits, dim=-1) * torch.log_softmax(policy_logits, dim=-1)).sum(-1)
        return log_probs, entropy, value


def create_test_config(**overrides):
    """Create a TrainingConfig for testing with optional overrides."""
    config_dict = {
        'enable_torch_compile': True,
        'torch_compile_mode': 'default',
        'torch_compile_dynamic': None,
        'torch_compile_fullgraph': False,
        'torch_compile_backend': None,
        'enable_compilation_fallback': True,
        'validate_compiled_output': True,
        'compilation_validation_tolerance': 1e-5,
        'compilation_warmup_steps': 2,  # Reduced for testing
        'enable_compilation_benchmarking': True,
        **overrides
    }
    return TrainingConfig(**config_dict)


@pytest.fixture
def sample_model():
    """Create a sample model for testing."""
    return MockActorCritic()


@pytest.fixture
def sample_input():
    """Create sample input tensor."""
    return torch.randn(1, 46, 9, 9)


@pytest.fixture
def training_config():
    """Create training configuration for testing."""
    return create_test_config()


class TestPerformanceBenchmarker:
    """Test the performance benchmarking framework."""
    
    def test_benchmarker_creation(self, training_config):
        """Test creating a benchmarker from config."""
        benchmarker = create_benchmarker(training_config)
        assert isinstance(benchmarker, PerformanceBenchmarker)
        assert benchmarker.warmup_iterations == 2  # From config
        assert benchmarker.benchmark_iterations == 100
    
    def test_model_benchmarking(self, sample_model, sample_input, training_config):
        """Test benchmarking a model."""
        benchmarker = create_benchmarker(training_config)
        benchmarker.benchmark_iterations = 5  # Reduced for testing
        
        result = benchmarker.benchmark_model(
            model=sample_model,
            input_tensor=sample_input,
            name="test_model"
        )
        
        assert result.name == "test_model"
        assert result.mean_time_ms > 0
        assert result.num_iterations == 5
        assert result.device == "cpu"
    
    def test_model_comparison(self, sample_input, training_config):
        """Test comparing two models."""
        model1 = MockActorCritic()
        model2 = MockActorCritic()
        
        benchmarker = create_benchmarker(training_config)
        benchmarker.benchmark_iterations = 3  # Reduced for testing
        
        comparison = benchmarker.compare_models(
            baseline_model=model1,
            optimized_model=model2,
            input_tensor=sample_input
        )
        
        assert comparison.baseline.name == "baseline"
        assert comparison.optimized.name == "optimized"
        assert comparison.speedup > 0
    
    def test_numerical_validation(self, sample_input, training_config):
        """Test numerical equivalence validation."""
        model1 = MockActorCritic()
        model2 = MockActorCritic()
        
        # Copy weights to ensure equivalence
        model2.load_state_dict(model1.state_dict())
        
        benchmarker = create_benchmarker(training_config)
        
        is_equivalent, max_diff, details = benchmarker.validate_numerical_equivalence(
            baseline_model=model1,
            optimized_model=model2,
            input_tensor=sample_input,
            tolerance=1e-5,
            num_samples=3
        )
        
        assert is_equivalent
        assert max_diff <= 1e-5
        assert 'max_policy_diff' in details
        assert 'max_value_diff' in details


class TestCompilationValidator:
    """Test the compilation validation framework."""
    
    def test_validator_creation(self, training_config):
        """Test creating a compilation validator."""
        validator = CompilationValidator(training_config)
        
        assert validator.enabled == True
        assert validator.mode == 'default'
        assert validator.enable_fallback == True
        assert validator.validate_output == True
    
    def test_compilation_disabled(self, sample_model, sample_input):
        """Test behavior when compilation is disabled."""
        config = create_test_config(enable_torch_compile=False)
        validator = CompilationValidator(config)
        
        result = validator.compile_model(sample_model, sample_input)
        
        assert result.success == True
        assert result.compiled_model is sample_model
        assert result.fallback_used == False
        assert 'compilation_skipped' in result.metadata
    
    @patch('torch.compile')
    def test_compilation_success(self, mock_compile, sample_model, sample_input, training_config):
        """Test successful compilation."""
        # Mock torch.compile to return the same model
        mock_compile.return_value = sample_model
        
        validator = CompilationValidator(training_config)
        validator.validate_output = False  # Skip validation for simplicity
        
        result = validator.compile_model(sample_model, sample_input)
        
        assert result.success == True
        assert result.compiled_model is not None
        assert result.fallback_used == False
        mock_compile.assert_called_once()
    
    @patch('torch.compile')
    def test_compilation_failure_with_fallback(self, mock_compile, sample_model, sample_input):
        """Test compilation failure with fallback enabled."""
        mock_compile.side_effect = RuntimeError("Compilation failed")
        
        config = create_test_config(enable_compilation_fallback=True)
        validator = CompilationValidator(config)
        
        result = validator.compile_model(sample_model, sample_input)
        
        assert result.success == False
        assert result.compiled_model is sample_model  # Fallback to original
        assert result.fallback_used == True
        assert "Compilation failed" in result.error_message
    
    @patch('torch.compile')
    def test_compilation_failure_without_fallback(self, mock_compile, sample_model, sample_input):
        """Test compilation failure without fallback."""
        mock_compile.side_effect = RuntimeError("Compilation failed")
        
        config = create_test_config(enable_compilation_fallback=False)
        validator = CompilationValidator(config)
        
        with pytest.raises(RuntimeError, match="Compilation failed"):
            validator.compile_model(sample_model, sample_input)
    
    def test_safe_compile_model_function(self, sample_model, sample_input, training_config):
        """Test the safe_compile_model convenience function."""
        with patch('torch.compile') as mock_compile:
            mock_compile.return_value = sample_model
            
            compiled_model, result = safe_compile_model(
                model=sample_model,
                sample_input=sample_input,
                config_training=training_config,
                model_name="test_model"
            )
            
            assert compiled_model is not None
            assert isinstance(result.success, bool)


class TestRealModelIntegration:
    """Test with real ResNet model architecture."""
    
    @pytest.fixture
    def resnet_model(self):
        """Create a real ResNet model for testing."""
        return ActorCriticResTower(
            input_channels=46,
            num_actions_total=13527,
            tower_depth=3,  # Reduced for testing
            tower_width=64,  # Reduced for testing
            se_ratio=0.25
        )
    
    def test_resnet_benchmarking(self, resnet_model, sample_input, training_config):
        """Test benchmarking with real ResNet model."""
        benchmarker = create_benchmarker(training_config)
        benchmarker.benchmark_iterations = 3  # Reduced for testing
        
        result = benchmarker.benchmark_model(
            model=resnet_model,
            input_tensor=sample_input,
            name="resnet_model",
            model_type="resnet"
        )
        
        assert result.name == "resnet_model"
        assert result.model_type == "resnet"
        assert result.mean_time_ms > 0
    
    @patch('torch.compile')
    def test_resnet_compilation(self, mock_compile, resnet_model, sample_input, training_config):
        """Test compilation with real ResNet model."""
        mock_compile.return_value = resnet_model
        
        validator = CompilationValidator(training_config)
        validator.validate_output = False  # Skip for simplicity
        
        result = validator.compile_model(resnet_model, sample_input, "resnet_test")
        
        assert result.success == True
        mock_compile.assert_called_once()


@pytest.mark.skipif(not hasattr(torch, 'compile'), reason="torch.compile not available")
class TestActualTorchCompile:
    """Test with actual torch.compile (if available)."""
    
    def test_actual_compilation(self, sample_model, sample_input):
        """Test actual torch.compile if available."""
        config = create_test_config(
            validate_compiled_output=True,
            compilation_validation_tolerance=1e-4  # Relaxed for actual compilation
        )
        
        validator = CompilationValidator(config)
        
        # This might fail on some systems/PyTorch versions, but should fallback gracefully
        result = validator.compile_model(sample_model, sample_input)
        
        # Should either succeed or fallback gracefully
        assert result.compiled_model is not None
        if not result.success:
            assert result.fallback_used == True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
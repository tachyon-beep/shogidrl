"""
test_env_manager.py: Comprehensive unit tests for EnvManager class.

Tests cover environment initialization, game setup, policy mapper configuration,
action space validation, seeding, and environment validation functionality.
"""

from unittest.mock import Mock, patch

import pytest

from keisei.config_schema import (
    AppConfig,
    EnvConfig,
    EvaluationConfig,
    LoggingConfig,
    ParallelConfig,
    TrainingConfig,
    WandBConfig,
    DisplayConfig,
)
from keisei.training.env_manager import EnvManager


@pytest.fixture
def mock_config():
    """Create a mock AppConfig for testing."""
    return AppConfig(
        parallel=ParallelConfig(
            enabled=False,
            num_workers=1,
            batch_size=32,
            sync_interval=100,
            compression_enabled=True,
            timeout_seconds=10.0,
            max_queue_size=1000,
            worker_seed_offset=1000,
        ),
        env=EnvConfig(
            device="cpu",
            num_actions_total=13527,
            input_channels=46,
            seed=42,
            max_moves_per_game=500,
        ),
        training=TrainingConfig(
            input_features="core46",
            model_type="resnet",
            total_timesteps=1000,
            steps_per_epoch=100,
            ppo_epochs=4,
            minibatch_size=32,
            learning_rate=0.001,
            gamma=0.99,
            clip_epsilon=0.2,
            value_loss_coeff=0.5,
            entropy_coef=0.01,
            render_every_steps=100,
            refresh_per_second=10,
            enable_spinner=True,
            tower_depth=9,
            tower_width=256,
            se_ratio=0.25,
            mixed_precision=False,
            ddp=False,  # Added default
            gradient_clip_max_norm=0.5,  # Added default
            lambda_gae=0.95,  # Added default
            checkpoint_interval_timesteps=10000,  # Added default
            evaluation_interval_timesteps=50000,  # Added default
            weight_decay=0.0,  # Added default
            normalize_advantages=True,
            lr_schedule_type=None,
            lr_schedule_kwargs=None,
            lr_schedule_step_on="epoch",
        ),
        evaluation=EvaluationConfig(
            enable_periodic_evaluation=True,
            num_games=1,  # Reduced for faster testing
            opponent_type="random",
            evaluation_interval_timesteps=50000,  # Added default
            max_moves_per_game=500,
            log_file_path_eval="eval_log.txt",
            wandb_log_eval=False,
        ),
        logging=LoggingConfig(
            log_file="logs/test_log.txt",
            model_dir="models/test_models/",
            run_name="test_run",
        ),
        wandb=WandBConfig(
            enabled=False,
            project="keisei-shogi-rl",  # Added default
            entity=None,  # Added default
            run_name_prefix="keisei",  # Added default
            watch_model=True,  # Added default
            watch_log_freq=1000,  # Added default
            watch_log_type="all",  # Added default
            log_model_artifact=False,
        ),
        display=DisplayConfig(display_moves=False, turn_tick=0.5),  # Added default
    )


@pytest.fixture
def logger_func():
    """Create a mock logger function."""
    return Mock()


class TestEnvManagerInitialization:
    """Test EnvManager initialization and setup."""

    @patch("keisei.training.env_manager.ShogiGame")
    @patch("keisei.training.env_manager.PolicyOutputMapper")
    def test_initialization_success(
        self, mock_policy_mapper_class, mock_shogi_game_class, mock_config, logger_func
    ):
        """Test successful EnvManager initialization."""
        # Setup mocks
        mock_game = Mock()
        mock_game.seed = Mock()
        mock_shogi_game_class.return_value = mock_game

        mock_mapper = Mock()
        mock_mapper.get_total_actions.return_value = 13527
        mock_policy_mapper_class.return_value = mock_mapper

        # Create EnvManager
        env_manager = EnvManager(mock_config, logger_func)
        env_manager.setup_environment()  # Call setup_environment

        # Verify initialization
        assert env_manager.config == mock_config
        assert env_manager.logger_func == logger_func
        assert env_manager.game == mock_game
        assert env_manager.policy_output_mapper == mock_mapper
        assert env_manager.action_space_size == 13527
        assert env_manager.obs_space_shape == (46, 9, 9)

        # Verify game seeding was called
        mock_game.seed.assert_called_once_with(42)
        logger_func.assert_any_call("Environment seeded with: 42")

    @patch("keisei.training.env_manager.ShogiGame")
    @patch("keisei.training.env_manager.PolicyOutputMapper")
    def test_initialization_no_seed(
        self, mock_policy_mapper_class, mock_shogi_game_class, mock_config, logger_func
    ):
        """Test initialization when no seed is specified."""
        # Remove seed from config
        mock_config.env.seed = None

        # Setup mocks
        mock_game = Mock()
        mock_game.seed = Mock()
        mock_shogi_game_class.return_value = mock_game

        mock_mapper = Mock()
        mock_mapper.get_total_actions.return_value = 13527
        mock_policy_mapper_class.return_value = mock_mapper

        # Create EnvManager
        env_manager = EnvManager(mock_config, logger_func)
        env_manager.setup_environment()  # Call setup_environment

        # Verify seeding was not called
        mock_game.seed.assert_not_called()

    @patch("keisei.training.env_manager.ShogiGame")
    def test_initialization_game_error(
        self, mock_shogi_game_class, mock_config, logger_func
    ):
        """Test initialization when ShogiGame creation fails."""
        # Setup mock to raise exception
        mock_shogi_game_class.side_effect = RuntimeError("Game initialization failed")

        # Verify exception is raised
        with pytest.raises(RuntimeError, match="Failed to initialize ShogiGame"):
            env_manager = EnvManager(mock_config, logger_func)
            env_manager.setup_environment()  # Call setup_environment

        logger_func.assert_any_call(
            "Error initializing ShogiGame: Game initialization failed. Aborting."
        )

    @patch("keisei.training.env_manager.ShogiGame")
    @patch("keisei.training.env_manager.PolicyOutputMapper")
    def test_initialization_policy_mapper_error(
        self, mock_policy_mapper_class, mock_shogi_game_class, mock_config, logger_func
    ):
        """Test initialization when PolicyOutputMapper creation fails."""
        # Setup game mock
        mock_game = Mock()
        mock_shogi_game_class.return_value = mock_game

        # Setup policy mapper to raise exception
        mock_policy_mapper_class.side_effect = ValueError(
            "Policy mapper initialization failed"
        )

        # Verify exception is raised
        with pytest.raises(
            RuntimeError, match="Failed to initialize PolicyOutputMapper"
        ):
            env_manager = EnvManager(mock_config, logger_func)
            env_manager.setup_environment()  # Call setup_environment

        logger_func.assert_any_call(
            "Error initializing PolicyOutputMapper: Policy mapper initialization failed"
        )


class TestEnvManagerActionSpaceValidation:
    """Test action space validation functionality."""

    @patch("keisei.training.env_manager.ShogiGame")
    @patch("keisei.training.env_manager.PolicyOutputMapper")
    def test_action_space_validation_success(
        self, mock_policy_mapper_class, mock_shogi_game_class, mock_config, logger_func
    ):
        """Test successful action space validation."""
        # Setup mocks
        mock_game = Mock()
        mock_shogi_game_class.return_value = mock_game

        mock_mapper = Mock()
        mock_mapper.get_total_actions.return_value = 13527  # Matches config
        mock_policy_mapper_class.return_value = mock_mapper

        # Create EnvManager
        env_manager = EnvManager(mock_config, logger_func)
        env_manager.setup_environment()  # Call setup_environment

        # Verify validation passed
        logger_func.assert_any_call("Action space validated: 13527 total actions")

    @patch("keisei.training.env_manager.ShogiGame")
    @patch("keisei.training.env_manager.PolicyOutputMapper")
    def test_action_space_validation_mismatch(
        self, mock_policy_mapper_class, mock_shogi_game_class, mock_config, logger_func
    ):
        """Test action space validation with mismatch."""
        # Setup mocks
        mock_game = Mock()
        mock_shogi_game_class.return_value = mock_game

        mock_mapper = Mock()
        mock_mapper.get_total_actions.return_value = 10000  # Different from config
        mock_policy_mapper_class.return_value = mock_mapper

        # Verify exception is raised
        with pytest.raises(
            RuntimeError, match="Failed to initialize PolicyOutputMapper"
        ):
            env_manager = EnvManager(mock_config, logger_func)
            env_manager.setup_environment()  # Call setup_environment

        expected_error = (
            "Action space mismatch: config specifies 13527 "
            "actions but PolicyOutputMapper provides 10000 actions"
        )
        logger_func.assert_any_call(f"CRITICAL: {expected_error}")


class TestEnvManagerEnvironmentOperations:
    """Test environment operation methods."""

    @pytest.mark.parametrize(
        "operation_name,setup_method,expected_result,expected_log_content",
        [
            ("reset_game", "setup_reset_success", True, None),
            (
                "reset_game",
                "setup_reset_failure",
                False,
                "Error resetting game: Reset failed",
            ),
            ("get_legal_moves_count", "setup_legal_moves_success", 3, None),
            (
                "get_legal_moves_count",
                "setup_legal_moves_failure",
                0,
                "Error getting legal moves count: Legal moves error",
            ),
        ],
        ids=[
            "reset_success",
            "reset_failure",
            "legal_moves_success",
            "legal_moves_failure",
        ],
    )
    @patch("keisei.training.env_manager.ShogiGame")
    @patch("keisei.training.env_manager.PolicyOutputMapper")
    def test_environment_operations(
        self,
        mock_policy_mapper_class,
        mock_shogi_game_class,
        mock_config,
        logger_func,
        operation_name,
        setup_method,
        expected_result,
        expected_log_content,
    ):
        """Test environment operations with success and failure scenarios."""
        # Setup mocks based on test scenario
        mock_game = Mock()
        mock_shogi_game_class.return_value = mock_game

        mock_mapper = Mock()
        mock_mapper.get_total_actions.return_value = 13527
        mock_policy_mapper_class.return_value = mock_mapper

        # Configure mock behavior based on setup method
        if setup_method == "setup_reset_success":
            mock_game.reset = Mock()
        elif setup_method == "setup_reset_failure":
            mock_game.reset = Mock(side_effect=Exception("Reset failed"))
        elif setup_method == "setup_legal_moves_success":
            mock_legal_moves = [Mock(), Mock(), Mock()]  # 3 legal moves
            mock_game.get_legal_moves = Mock(return_value=mock_legal_moves)
        elif setup_method == "setup_legal_moves_failure":
            mock_game.get_legal_moves = Mock(side_effect=Exception("Legal moves error"))

        # Create EnvManager
        env_manager = EnvManager(mock_config, logger_func)
        env_manager.setup_environment()  # Call setup_environment

        # Test the operation
        operation_method = getattr(env_manager, operation_name)
        result = operation_method()

        # Verify result
        assert result == expected_result

        # Verify logging if expected
        if expected_log_content:
            logger_func.assert_any_call(expected_log_content)


class TestEnvManagerSeeding:
    """Test environment seeding functionality."""

    @patch("keisei.training.env_manager.ShogiGame")
    @patch("keisei.training.env_manager.PolicyOutputMapper")
    def test_setup_seeding_with_explicit_seed(
        self, mock_policy_mapper_class, mock_shogi_game_class, mock_config, logger_func
    ):
        """Test seeding with explicit seed value."""
        # Setup mocks
        mock_game = Mock()
        mock_game.seed = Mock()
        mock_shogi_game_class.return_value = mock_game

        mock_mapper = Mock()
        mock_mapper.get_total_actions.return_value = 13527
        mock_policy_mapper_class.return_value = mock_mapper

        # Create EnvManager
        env_manager = EnvManager(mock_config, logger_func)
        env_manager.setup_environment()  # Call setup_environment

        # Reset logger_func to track new calls
        logger_func.reset_mock()

        # Test seeding with explicit value
        result = env_manager.setup_seeding(123)

        # Verify seeding
        assert result is True
        mock_game.seed.assert_called_with(
            123
        )  # Should use explicit seed, not config seed
        logger_func.assert_any_call("Environment re-seeded with: 123")

    @patch("keisei.training.env_manager.ShogiGame")
    @patch("keisei.training.env_manager.PolicyOutputMapper")
    def test_setup_seeding_with_config_seed(
        self, mock_policy_mapper_class, mock_shogi_game_class, mock_config, logger_func
    ):
        """Test seeding with config seed value."""
        # Setup mocks
        mock_game = Mock()
        mock_game.seed = Mock()
        mock_shogi_game_class.return_value = mock_game

        mock_mapper = Mock()
        mock_mapper.get_total_actions.return_value = 13527
        mock_policy_mapper_class.return_value = mock_mapper

        # Create EnvManager
        env_manager = EnvManager(mock_config, logger_func)
        env_manager.setup_environment()  # Call setup_environment

        # Reset logger_func to track new calls
        logger_func.reset_mock()

        # Test seeding with None (should use config seed)
        result = env_manager.setup_seeding(None)  # type: ignore

        # Verify seeding used config value
        assert result is True
        mock_game.seed.assert_called_with(42)  # Should use config seed
        logger_func.assert_any_call("Environment re-seeded with: 42")

    @patch("keisei.training.env_manager.ShogiGame")
    @patch("keisei.training.env_manager.PolicyOutputMapper")
    def test_setup_seeding_no_seed_method(
        self, mock_policy_mapper_class, mock_shogi_game_class, mock_config, logger_func
    ):
        """Test seeding when game has no seed method."""
        # Setup mocks - game without seed method
        mock_game = Mock(spec=[])  # Empty spec means no seed method
        mock_shogi_game_class.return_value = mock_game

        mock_mapper = Mock()
        mock_mapper.get_total_actions.return_value = 13527
        mock_policy_mapper_class.return_value = mock_mapper

        # Create EnvManager
        env_manager = EnvManager(mock_config, logger_func)
        env_manager.setup_environment()  # Call setup_environment

        # Test seeding
        result = env_manager.setup_seeding(123)

        # Verify seeding was not attempted
        assert result is False

    @patch("keisei.training.env_manager.ShogiGame")
    @patch("keisei.training.env_manager.PolicyOutputMapper")
    def test_setup_seeding_error(
        self, mock_policy_mapper_class, mock_shogi_game_class, mock_config, logger_func
    ):
        """Test seeding when seed method raises error."""
        # Setup mocks
        mock_game = Mock()
        mock_game.seed.side_effect = Exception("Seeding failed")
        mock_shogi_game_class.return_value = mock_game

        mock_mapper = Mock()
        mock_mapper.get_total_actions.return_value = 13527
        mock_policy_mapper_class.return_value = mock_mapper

        # Create EnvManager
        env_manager = EnvManager(mock_config, logger_func)
        env_manager.setup_environment()  # Call setup_environment

        # Reset logger_func to track new calls
        logger_func.reset_mock()

        # Test seeding
        result = env_manager.setup_seeding(123)

        # Verify error handling
        assert result is False
        logger_func.assert_any_call("Error setting environment seed: Seeding failed")


class TestEnvManagerValidation:
    """Test environment validation functionality."""

    @patch("keisei.training.env_manager.ShogiGame")
    @patch("keisei.training.env_manager.PolicyOutputMapper")
    def test_validate_environment_success(
        self, mock_policy_mapper_class, mock_shogi_game_class, mock_config, logger_func
    ):
        """Test successful environment validation."""
        # Setup mocks
        mock_game = Mock()
        mock_game.get_board_state_copy.return_value = Mock()
        mock_game.reset = Mock()
        mock_shogi_game_class.return_value = mock_game

        mock_mapper = Mock()
        mock_mapper.get_total_actions.return_value = 13527
        mock_policy_mapper_class.return_value = mock_mapper

        # Create EnvManager
        env_manager = EnvManager(mock_config, logger_func)
        env_manager.setup_environment()  # Call setup_environment

        # Test validation
        result = env_manager.validate_environment()

        # Verify validation passed
        assert result is True
        logger_func.assert_any_call("Environment validation passed")

    @patch("keisei.training.env_manager.ShogiGame")
    @patch("keisei.training.env_manager.PolicyOutputMapper")
    def test_validate_environment_game_none(
        self, mock_policy_mapper_class, mock_shogi_game_class, mock_config, logger_func
    ):
        """Test validation when game is None."""
        # Setup mocks
        mock_game = Mock()
        mock_shogi_game_class.return_value = mock_game

        mock_mapper = Mock()
        mock_mapper.get_total_actions.return_value = 13527
        mock_policy_mapper_class.return_value = mock_mapper

        # Create EnvManager and manually set game to None
        env_manager = EnvManager(mock_config, logger_func)
        env_manager.game = None  # type: ignore

        # Test validation
        result = env_manager.validate_environment()

        # Verify validation failed
        assert result is False
        logger_func.assert_any_call(
            "Environment validation failed: game not initialized"
        )

    @patch("keisei.training.env_manager.ShogiGame")
    @patch("keisei.training.env_manager.PolicyOutputMapper")
    def test_validate_environment_policy_mapper_none(
        self, mock_policy_mapper_class, mock_shogi_game_class, mock_config, logger_func
    ):
        """Test validation when policy mapper is None."""
        # Setup mocks
        mock_game = Mock()
        mock_shogi_game_class.return_value = mock_game

        mock_mapper = Mock()
        mock_mapper.get_total_actions.return_value = 13527
        mock_policy_mapper_class.return_value = mock_mapper

        # Create EnvManager and manually set policy mapper to None
        env_manager = EnvManager(mock_config, logger_func)
        env_manager.setup_environment()  # Call setup
        env_manager.policy_output_mapper = None  # type: ignore

        # Test validation
        result = env_manager.validate_environment()

        # Verify validation failed
        assert result is False
        logger_func.assert_any_call(
            "Environment validation failed: policy mapper not initialized"
        )

    @patch("keisei.training.env_manager.ShogiGame")
    @patch("keisei.training.env_manager.PolicyOutputMapper")
    def test_validate_environment_invalid_action_space(
        self, mock_policy_mapper_class, mock_shogi_game_class, mock_config, logger_func
    ):
        """Test validation with invalid action space."""
        # Setup mocks
        mock_game = Mock()
        mock_shogi_game_class.return_value = mock_game

        mock_mapper = Mock()
        mock_mapper.get_total_actions.return_value = 13527
        mock_policy_mapper_class.return_value = mock_mapper

        # Create EnvManager and manually set invalid action space size
        env_manager = EnvManager(mock_config, logger_func)
        env_manager.setup_environment()  # Call setup
        env_manager.action_space_size = 0

        # Test validation
        result = env_manager.validate_environment()

        # Verify validation failed
        assert result is False
        logger_func.assert_any_call(
            "Environment validation failed: invalid action space size"
        )


class TestEnvManagerUtilities:
    """Test utility methods and information retrieval."""

    @patch("keisei.training.env_manager.ShogiGame")
    @patch("keisei.training.env_manager.PolicyOutputMapper")
    def test_get_environment_info(
        self, mock_policy_mapper_class, mock_shogi_game_class, mock_config, logger_func
    ):
        """Test environment information retrieval."""
        # Setup mocks
        mock_game = Mock()
        mock_shogi_game_class.return_value = mock_game

        mock_mapper = Mock()
        mock_mapper.get_total_actions.return_value = 13527
        mock_policy_mapper_class.return_value = mock_mapper

        # Create EnvManager
        env_manager = EnvManager(mock_config, logger_func)
        env_manager.setup_environment()  # Call setup

        # Get environment info
        info = env_manager.get_environment_info()

        # Verify information
        expected_info = {
            "game_type": "Mock",  # Mock class name
            "action_space_size": 13527,
            "obs_space_shape": (46, 9, 9),
            "input_channels": 46,
            "num_actions_total": 13527,
            "seed": 42,
            "policy_mapper_type": "Mock",  # Mock class name
            "game": mock_game,  # Actual game object
            "policy_mapper": mock_mapper,  # Actual policy mapper object
        }
        assert info == expected_info

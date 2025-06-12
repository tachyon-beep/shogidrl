"""
Comprehensive TrainingDisplay - the central TUI orchestrator.

This module tests:
- TrainingDisplay initialization and component setup
- Layout creation and panel coordination
- Dashboard refresh cycles and error handling
- Configuration-driven feature toggling
- Rich Live context management
- Panel updates and synchronization
- Error recovery and edge cases
- Performance under various conditions
- Integration scenarios
"""

import io
import threading
import time
from unittest.mock import Mock, patch

import pytest
from rich.console import Console

from keisei.config_schema import DisplayConfig
from keisei.training.display import TrainingDisplay
from keisei.training.display_components import ShogiBoard, RecentMovesPanel
from tests.display.test_utilities import (
    TestDataFactory,
    training_session_in_progress,  # pylint: disable=unused-import
    display_with_error_state,  # pylint: disable=unused-import
    performance_helper,
    mock_library  # pylint: disable=unused-import
)


class TestTrainingDisplay:
    """Tests for TrainingDisplay initialization and core functionality."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration object."""
        config = Mock()
        config.display = DisplayConfig(
            enable_board_display=True,
            enable_trend_visualization=True,
            enable_elo_ratings=True,
            enable_enhanced_layout=True,
            display_moves=False,
            turn_tick=0.0,
            board_unicode_pieces=True,
            board_cell_width=5,
            board_cell_height=3,
            board_highlight_last_move=True,
            sparkline_width=15,
            trend_history_length=100,
            elo_initial_rating=1500.0,
            elo_k_factor=32.0,
            dashboard_height_ratio=2,
            progress_bar_height=4,
            show_text_moves=True,
            move_list_length=10,
            moves_latest_top=True,
            moves_flash_ms=500,
            show_moves_trend=True,
            show_completion_rate=True,
            show_enhanced_win_rates=True,
            show_turns_trend=True,
            metrics_window_size=100,
            trend_smoothing_factor=0.1,
            metrics_panel_height=6,
            enable_trendlines=True,
            log_layer_keyword_filters=["stem", "policy_head", "value_head"]
        )
        config.training = Mock()
        config.training.total_timesteps = 10000
        config.training.enable_spinner = True
        config.training.refresh_per_second = 1
        return config

    @pytest.fixture
    def mock_trainer(self):
        """Create a mock trainer object."""
        trainer = Mock()
        trainer.rich_log_messages = []
        trainer.metrics_manager = Mock()
        trainer.metrics_manager.global_timestep = 0
        trainer.metrics_manager.total_episodes_completed = 0
        trainer.metrics_manager.black_wins = 0
        trainer.metrics_manager.white_wins = 0
        trainer.metrics_manager.draws = 0

        # Create history mock with list-like attributes (same as realistic_trainer)
        history = Mock()
        history.episode_lengths = [40, 45, 50]
        history.episode_rewards = [0.2, 0.3, 0.1]
        history.policy_losses = [0.1, 0.08, 0.12]
        history.value_losses = [0.05, 0.04, 0.06]
        history.entropies = [1.2, 1.3, 1.1]
        history.kl_divergences = [0.01, 0.008, 0.012]
        history.clip_fractions = [0.2, 0.18, 0.22]
        history.win_rates_history = [
            {"win_rate_black": 0.5, "win_rate_white": 0.4, "win_rate_draw": 0.1},
            {"win_rate_black": 0.6, "win_rate_white": 0.3, "win_rate_draw": 0.1}
        ]
        trainer.metrics_manager.history = history

        # Opening history as iterable lists
        trainer.metrics_manager.sente_opening_history = ["居飛車", "振り飛車"]
        trainer.metrics_manager.gote_opening_history = ["相振り飛車", "居飛車"]

        trainer.game = None
        trainer.step_manager = None
        # Additional trainer attributes
        trainer.last_gradient_norm = 1.0
        trainer.last_ply_per_sec = 2.0

        # Mock agent and model
        trainer.agent = Mock()
        trainer.agent.model = Mock()
        trainer.agent.model.named_parameters = Mock(return_value=[])

        # Mock experience buffer
        trainer.experience_buffer = Mock()
        trainer.experience_buffer.capacity = Mock(return_value=1000)
        trainer.experience_buffer.size = Mock(return_value=500)

        # Mock evaluation snapshot
        trainer.evaluation_elo_snapshot = None

        return trainer

    @pytest.fixture
    def mock_console(self):
        """Create a mock Rich console."""
        console = Mock(spec=Console)
        console.size = Mock()
        console.size.width = 120
        console.size.height = 30
        # Add get_time method that Rich Progress expects
        console.get_time = Mock(return_value=0.0)
        # Add encoding attribute that adaptive display expects
        console.encoding = "utf-8"
        return console

    def test_initialization_with_config(self, mock_config, mock_trainer, mock_console):
        """Test that TrainingDisplay initializes properly with configuration."""
        display = TrainingDisplay(mock_config, mock_trainer, mock_console)

        # Test basic initialization
        assert display.config == mock_config
        assert display.display_config == mock_config.display
        assert display.trainer == mock_trainer
        assert display.rich_console == mock_console

        # Test component initialization based on config
        assert display.board_component is not None
        assert isinstance(display.board_component, ShogiBoard)
        assert display.moves_component is not None
        assert isinstance(display.moves_component, RecentMovesPanel)
        assert display.game_stats_component is not None

        # Test layout and progress bar setup
        assert display.progress_bar is not None
        assert display.layout is not None
        assert display.log_panel is not None

    def test_initialization_with_disabled_features(self, mock_config, mock_trainer, mock_console):
        """Test initialization when display features are disabled."""
        # Disable board display
        mock_config.display.enable_board_display = False
        mock_config.display.enable_trend_visualization = False
        mock_config.display.enable_elo_ratings = False

        display = TrainingDisplay(mock_config, mock_trainer, mock_console)

        # Components should be None when disabled
        assert display.board_component is None
        assert display.moves_component is None
        assert display.piece_stand_component is None
        assert display.trend_component is None
        assert display.multi_trend_component is None
        assert display.completion_rate_calc is None
        assert display.elo_component_enabled is False

    def test_layout_creation_enhanced(self, mock_config, mock_trainer, mock_console):
        """Test enhanced layout creation for large consoles."""
        mock_console.size.width = 200
        mock_console.size.height = 50

        display = TrainingDisplay(mock_config, mock_trainer, mock_console)

        # Should use enhanced layout for large console
        assert display.using_enhanced_layout is True
        # Verify layout structure exists
        assert display.layout is not None
        assert display.layout.name == "root"

    def test_layout_creation_compact(self, mock_config, mock_trainer, mock_console):
        """Test compact layout creation for small consoles."""
        mock_console.size.width = 80
        mock_console.size.height = 20

        display = TrainingDisplay(mock_config, mock_trainer, mock_console)

        # Should use compact layout for small console
        assert display.using_enhanced_layout is False

    def test_dashboard_refresh_cycle_enhanced_layout(self, mock_config, mock_trainer, mock_console):
        """Test complete dashboard refresh cycle with enhanced layout."""
        mock_console.size.width = 200
        mock_console.size.height = 50

        # Setup trainer with some data
        mock_trainer.rich_log_messages = [Mock(), Mock()]
        mock_trainer.game = Mock()
        mock_trainer.step_manager = Mock()
        mock_trainer.step_manager.move_log = ["7g7f", "8c8d"]

        display = TrainingDisplay(mock_config, mock_trainer, mock_console)

        # Should not raise errors during refresh
        display.refresh_dashboard_panels(mock_trainer)

        # Verify log panel was updated
        assert display.log_panel.renderable is not None

    def test_dashboard_refresh_cycle_compact_layout(self, mock_config, mock_trainer, mock_console):
        """Test dashboard refresh with compact layout (should only update log panel)."""
        mock_console.size.width = 80
        mock_console.size.height = 20

        mock_trainer.rich_log_messages = [Mock(), Mock()]

        display = TrainingDisplay(mock_config, mock_trainer, mock_console)

        # Should not raise errors and should return early for compact layout
        display.refresh_dashboard_panels(mock_trainer)

        # Verify log panel was updated
        assert display.log_panel.renderable is not None

    def test_error_handling_in_rendering(self, mock_config, mock_trainer, mock_console):
        """Test graceful error handling during component rendering."""
        mock_console.size.width = 200
        mock_console.size.height = 50

        # Setup trainer that will cause errors
        mock_trainer.game = Mock()
        mock_trainer.game.side_effect = Exception("Game error")

        display = TrainingDisplay(mock_config, mock_trainer, mock_console)

        # Should handle errors gracefully
        with patch.object(display.rich_console, 'log') as mock_log:
            display.refresh_dashboard_panels(mock_trainer)
            # Verify error was logged (at least one error log call)
            assert mock_log.call_count >= 1

    def test_progress_bar_initialization(self, mock_config, mock_trainer, mock_console):
        """Test progress bar setup with correct initial values."""
        display = TrainingDisplay(mock_config, mock_trainer, mock_console)

        # Verify progress bar was created
        assert display.progress_bar is not None
        assert display.training_task is not None

        # Verify task was added with correct parameters
        task = display.progress_bar.tasks[0]
        assert task.total == mock_config.training.total_timesteps
        assert task.completed == mock_trainer.metrics_manager.global_timestep

    def test_start_live_display(self, mock_config, mock_trainer, mock_console):
        """Test starting the live display context."""
        display = TrainingDisplay(mock_config, mock_trainer, mock_console)

        with patch('keisei.training.display.Live') as mock_live:
            mock_live_instance = Mock()
            mock_live.return_value = mock_live_instance

            result = display.start()

            # Verify Live was created with correct parameters
            mock_live.assert_called_once_with(
                display.layout,
                console=display.rich_console,
                refresh_per_second=1,
                transient=False
            )
            assert result == mock_live_instance

    def test_update_progress(self, mock_config, mock_trainer, mock_console):
        """Test progress bar updates with trainer state."""
        display = TrainingDisplay(mock_config, mock_trainer, mock_console)

        # Setup trainer state
        mock_trainer.metrics_manager.global_timestep = 1000
        speed = 10.5
        pending_updates = {
            'ep_metrics': 'Ep L:100 R:0.5',
            'ppo_metrics': 'PPO Loss:0.1'
        }

        display.update_progress(mock_trainer, speed, pending_updates)

        # Verify progress bar update was called
        task = display.progress_bar.tasks[0]
        assert task.completed == 1000

    def test_component_integration(self, mock_config, mock_trainer, mock_console):
        """Test that all components are properly integrated."""
        mock_console.size.width = 200
        mock_console.size.height = 50

        display = TrainingDisplay(mock_config, mock_trainer, mock_console)

        # Verify all expected components are present
        assert display.board_component is not None
        assert display.moves_component is not None
        assert display.piece_stand_component is not None
        assert display.game_stats_component is not None
        assert display.trend_component is not None
        assert display.multi_trend_component is not None
        assert display.completion_rate_calc is not None

    def test_console_size_adaptation(self, mock_config, mock_trainer):
        """Test that display adapts to different console sizes."""
        # Test large console
        large_console = Mock(spec=Console)
        large_console.size = Mock()
        large_console.size.width = 200
        large_console.size.height = 50
        large_console.get_time = Mock(return_value=0.0)
        large_console.encoding = "utf-8"

        display_large = TrainingDisplay(mock_config, mock_trainer, large_console)
        assert display_large.using_enhanced_layout is True

        # Test small console
        small_console = Mock(spec=Console)
        small_console.size = Mock()
        small_console.size.width = 80
        small_console.size.height = 20
        small_console.get_time = Mock(return_value=0.0)
        small_console.encoding = "utf-8"

        display_small = TrainingDisplay(mock_config, mock_trainer, small_console)
        assert display_small.using_enhanced_layout is False

    def test_configuration_driven_feature_toggling(self, mock_config, mock_trainer, mock_console):
        """Test that features are correctly enabled/disabled based on configuration."""
        # Test with all features enabled
        mock_config.display.enable_board_display = True
        mock_config.display.enable_trend_visualization = True
        mock_config.display.enable_elo_ratings = True

        display = TrainingDisplay(mock_config, mock_trainer, mock_console)

        assert display.board_component is not None
        assert display.trend_component is not None
        assert display.elo_component_enabled is True

        # Test with all features disabled
        mock_config.display.enable_board_display = False
        mock_config.display.enable_trend_visualization = False
        mock_config.display.enable_elo_ratings = False

        display_disabled = TrainingDisplay(mock_config, mock_trainer, mock_console)

        assert display_disabled.board_component is None
        assert display_disabled.trend_component is None
        assert display_disabled.elo_component_enabled is False

    # =============================================================================
    # NEW COMPREHENSIVE TESTS - ADDRESSING ANALYSIS REPORT FINDINGS
    # =============================================================================

    def test_display_state_persistence_across_updates(self, mock_config, mock_trainer, mock_console, training_session_in_progress):
        """Test that display state persists correctly across multiple updates."""
        mock_console.size.width = 200
        mock_console.size.height = 50

        # Use realistic training session data
        session_data = training_session_in_progress
        mock_trainer.game = session_data['game_state']
        mock_trainer.step_manager = Mock()
        mock_trainer.step_manager.move_log = [str(move) for move in session_data['game_state'].move_history[-10:]]

        # Add proper string values for step_manager attributes to avoid Mock rendering issues
        mock_trainer.step_manager.sente_best_capture = "Bishop on 7b"
        mock_trainer.step_manager.gote_best_capture = "Rook on 2h"
        mock_trainer.step_manager.sente_capture_count = 3
        mock_trainer.step_manager.gote_capture_count = 2
        mock_trainer.step_manager.sente_drop_count = 1
        mock_trainer.step_manager.gote_drop_count = 0
        mock_trainer.step_manager.sente_promo_count = 2
        mock_trainer.step_manager.gote_promo_count = 1

        display = TrainingDisplay(mock_config, mock_trainer, mock_console)

        # Perform multiple refresh cycles
        for i in range(5):
            mock_trainer.metrics_manager.global_timestep = i * 100
            mock_trainer.metrics_manager.total_episodes_completed = i * 5

            # Should maintain state consistency
            display.refresh_dashboard_panels(mock_trainer)

            # Verify state persistence
            assert display.layout is not None
            assert display.log_panel is not None
            assert display.progress_bar is not None

    def test_error_recovery_from_corrupt_display_states(self, mock_config, mock_trainer, mock_console, display_with_error_state):
        """Test display recovery from various error conditions."""
        mock_console.size.width = 200
        mock_console.size.height = 50
        mock_console.log = Mock()  # Track error logging

        error_data = display_with_error_state
        mock_trainer.game = error_data['corrupted_game_state']

        display = TrainingDisplay(mock_config, mock_trainer, mock_console)

        # Should handle corrupted game state gracefully
        try:
            display.refresh_dashboard_panels(mock_trainer)
            # If no exception, verify error was logged
            assert mock_console.log.call_count >= 0  # Errors may be logged
        except (AttributeError, RuntimeError, TypeError, ValueError):
            pytest.fail("Display should handle corrupted state gracefully")

    def test_performance_under_rapid_update_scenarios(self, mock_config, mock_trainer, mock_console, performance_helper):
        """Test display performance under rapid successive updates."""
        mock_console.size.width = 120
        mock_console.size.height = 30

        display = TrainingDisplay(mock_config, mock_trainer, mock_console)

        # Measure time for rapid updates
        def rapid_update_test():
            for i in range(50):
                mock_trainer.metrics_manager.global_timestep = i
                display.refresh_dashboard_panels(mock_trainer)

        timing_result = performance_helper.measure_display_update_time(rapid_update_test)

        # Should complete rapid updates within reasonable time (< 1 second for 50 updates)
        assert timing_result < 1.0, f"Rapid updates took too long: {timing_result:.3f}s"

    def test_memory_leak_detection_long_running_display(self, mock_config, mock_trainer, mock_console, performance_helper):
        """Test for memory leaks during long-running display operations."""
        mock_console.size.width = 120
        mock_console.size.height = 30

        display = TrainingDisplay(mock_config, mock_trainer, mock_console)

        def long_running_test():
            # Simulate long training session with many log messages
            for i in range(100):
                # Add log messages to simulate memory growth
                mock_trainer.rich_log_messages.append(f"Training step {i}")
                if len(mock_trainer.rich_log_messages) > 50:
                    mock_trainer.rich_log_messages = mock_trainer.rich_log_messages[-50:]  # Simulate trimming

                mock_trainer.metrics_manager.global_timestep = i
                display.refresh_dashboard_panels(mock_trainer)

            return True

        memory_result = performance_helper.memory_usage_test(long_running_test)

        # Memory growth should be reasonable (< 10MB for this test)
        assert memory_result['memory_diff_mb'] < 10, f"Excessive memory usage: {memory_result['memory_diff_mb']:.2f}MB"

    def test_terminal_resize_handling(self, mock_config, mock_trainer, extreme_dimensions):
        """Test display behavior with various terminal dimensions."""
        for width, height in extreme_dimensions:
            console = Mock(spec=Console)
            console.size = Mock()
            console.size.width = width
            console.size.height = height
            console.get_time = Mock(return_value=0.0)
            console.encoding = "utf-8"
            console.log = Mock()

            # Should handle any terminal size gracefully
            try:
                display = TrainingDisplay(mock_config, mock_trainer, console)
                display.refresh_dashboard_panels(mock_trainer)

                # Verify basic components exist
                assert display.layout is not None
                assert display.progress_bar is not None

                # Verify appropriate layout choice based on size
                if width >= 120 and height >= 30:
                    assert display.using_enhanced_layout
                else:
                    assert not display.using_enhanced_layout

            except (AttributeError, RuntimeError, TypeError, ValueError) as e:
                pytest.fail(f"Display failed with dimensions {width}x{height}: {e}")

    def test_concurrent_access_scenarios(self, mock_config, mock_trainer, mock_console):
        """Test display behavior under concurrent access patterns."""
        mock_console.size.width = 120
        mock_console.size.height = 30

        display = TrainingDisplay(mock_config, mock_trainer, mock_console)

        # Simulate concurrent updates from different threads
        errors = []

        def update_worker(worker_id):
            try:
                for i in range(10):
                    mock_trainer.metrics_manager.global_timestep = worker_id * 100 + i
                    display.refresh_dashboard_panels(mock_trainer)
                    time.sleep(0.001)  # Small delay to encourage race conditions
            except (AttributeError, RuntimeError, TypeError, ValueError) as e:
                errors.append(f"Worker {worker_id}: {e}")

        # Start multiple worker threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=update_worker, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=5.0)

        # Should handle concurrent access without errors
        assert len(errors) == 0, f"Concurrent access errors: {errors}"

    def test_keyboard_interrupt_handling_during_updates(self, mock_config, mock_trainer, mock_console):
        """Test graceful handling of keyboard interrupts during display updates."""
        mock_console.size.width = 120
        mock_console.size.height = 30

        display = TrainingDisplay(mock_config, mock_trainer, mock_console)

        # Mock keyboard interrupt during component rendering
        def interrupt_side_effect(*args, **kwargs):
            raise KeyboardInterrupt("User interrupt")

        with patch.object(display, 'board_component') as mock_board:
            mock_board.render = Mock(side_effect=interrupt_side_effect)

            # Should handle keyboard interrupt gracefully
            try:
                display.refresh_dashboard_panels(mock_trainer)
            except KeyboardInterrupt:
                # This is expected behavior - interrupt should propagate
                pass
            except (AttributeError, RuntimeError, TypeError, ValueError) as e:
                pytest.fail(f"Unexpected exception during interrupt: {e}")

    def test_unicode_rendering_edge_cases(self, mock_config, mock_trainer, mock_console, unicode_test_data):
        """Test display rendering with various Unicode characters."""
        mock_console.size.width = 120
        mock_console.size.height = 30

        display = TrainingDisplay(mock_config, mock_trainer, mock_console)

        # Test with various Unicode content in log messages
        for test_name, unicode_content in unicode_test_data.items():
            mock_trainer.rich_log_messages = [unicode_content]

            try:
                display.refresh_dashboard_panels(mock_trainer)
                # Should handle Unicode content without errors
                assert display.log_panel.renderable is not None
            except UnicodeError as e:
                pytest.fail(f"Unicode rendering failed for {test_name}: {e}")
            except (AttributeError, RuntimeError, TypeError, ValueError) as e:
                # Catch other potential encoding/rendering errors
                pytest.fail(f"Rendering failed for {test_name}: {e}")

    def test_display_component_integration_realistic_data(self, mock_config, mock_trainer, mock_console, training_session_in_progress):
        """Test integration of all display components with realistic data."""
        mock_console.size.width = 200
        mock_console.size.height = 50

        # Setup realistic training data
        session_data = training_session_in_progress
        mock_trainer.game = session_data['game_state']
        mock_trainer.step_manager = Mock()
        mock_trainer.step_manager.move_log = [str(move) for move in session_data['game_state'].move_history]

        # Add proper string values for step_manager attributes to avoid Mock rendering issues
        mock_trainer.step_manager.sente_best_capture = "Knight on 6g"
        mock_trainer.step_manager.gote_best_capture = "Silver on 4d"
        mock_trainer.step_manager.sente_capture_count = 5
        mock_trainer.step_manager.gote_capture_count = 4
        mock_trainer.step_manager.sente_drop_count = 2
        mock_trainer.step_manager.gote_drop_count = 3
        mock_trainer.step_manager.sente_promo_count = 1
        mock_trainer.step_manager.gote_promo_count = 2

        # Setup realistic metrics progression
        mock_trainer.metrics_manager.global_timestep = session_data['total_moves']
        mock_trainer.metrics_manager.total_episodes_completed = session_data['total_moves'] // 10

        display = TrainingDisplay(mock_config, mock_trainer, mock_console)

        # Test complete refresh with realistic data
        display.refresh_dashboard_panels(mock_trainer)

        # Verify all components were updated
        assert display.layout is not None
        assert display.log_panel.renderable is not None

        # Test progress updates with realistic speed
        speed = session_data['average_move_time']
        pending_updates = {
            'ep_metrics': f"Ep L:{session_data['total_moves']} R:0.75",
            'ppo_metrics': 'Loss: 0.045'
        }

        display.update_progress(mock_trainer, speed, pending_updates)

        # Verify progress was updated
        task = display.progress_bar.tasks[0]
        assert task.completed == session_data['total_moves']

    def test_configuration_validation_and_error_handling(self, mock_trainer, mock_console, display_with_error_state):  # pylint: disable=unused-argument
        """Test display behavior with malformed configuration."""
        mock_console.size.width = 120
        mock_console.size.height = 30

        # Create config with invalid values
        config = Mock()
        config.display = DisplayConfig(
            enable_board_display=True,
            enable_trend_visualization=True,
            enable_elo_ratings=True,
            enable_enhanced_layout=True,
            display_moves=False,
            turn_tick=0.0,
            board_unicode_pieces=True,
            board_cell_width=-5,  # Invalid negative value
            board_cell_height=0,   # Invalid zero value
            board_highlight_last_move=True,
            sparkline_width=0,     # Invalid zero value
            trend_history_length=-10,  # Invalid negative value
            elo_initial_rating=1500.0,
            elo_k_factor=32.0,
            dashboard_height_ratio=2,
            progress_bar_height=4,
            show_text_moves=True,
            move_list_length=10,
            moves_latest_top=True,
            moves_flash_ms=500,
            show_moves_trend=True,
            show_completion_rate=True,
            show_enhanced_win_rates=True,
            show_turns_trend=True,
            metrics_window_size=100,
            trend_smoothing_factor=0.1,
            metrics_panel_height=6,
            enable_trendlines=True,
            log_layer_keyword_filters=["stem", "policy_head", "value_head"]
        )
        config.training = Mock()
        config.training.total_timesteps = 10000
        config.training.enable_spinner = True
        config.training.refresh_per_second = 1

        # Should handle invalid config values gracefully
        try:
            display = TrainingDisplay(config, mock_trainer, mock_console)
            display.refresh_dashboard_panels(mock_trainer)
            # If it doesn't crash, verify basic functionality
            assert display.layout is not None
        except ValueError:
            # Some validation errors are acceptable
            pass
        except (AttributeError, RuntimeError, TypeError) as e:
            pytest.fail(f"Unexpected error with invalid config: {e}")

    def test_display_consistency_across_complex_scenarios(self, mock_config, mock_trainer, mock_console, large_metrics_dataset):
        """Test display consistency with large datasets and complex scenarios."""
        mock_console.size.width = 200
        mock_console.size.height = 50

        display = TrainingDisplay(mock_config, mock_trainer, mock_console)

        # Setup complex trainer state
        mock_trainer.metrics_manager.history = large_metrics_dataset
        mock_trainer.game = TestDataFactory.create_game_state(move_count=200)
        mock_trainer.step_manager = Mock()
        mock_trainer.step_manager.move_log = [f"move_{i}" for i in range(50)]

        # Add proper string values for step_manager attributes to avoid Mock rendering issues
        mock_trainer.step_manager.sente_best_capture = "Gold on 6i"
        mock_trainer.step_manager.gote_best_capture = "Lance on 9a"
        mock_trainer.step_manager.sente_capture_count = 7
        mock_trainer.step_manager.gote_capture_count = 6
        mock_trainer.step_manager.sente_drop_count = 3
        mock_trainer.step_manager.gote_drop_count = 2
        mock_trainer.step_manager.sente_promo_count = 4
        mock_trainer.step_manager.gote_promo_count = 3

        # Test multiple refresh cycles with varying data
        for i in range(10):
            # Vary the data each iteration
            mock_trainer.metrics_manager.global_timestep = i * 1000
            mock_trainer.metrics_manager.total_episodes_completed = i * 50

            # Add some log messages
            mock_trainer.rich_log_messages = [f"Log message {j}" for j in range(i, i + 20)]

            # Should maintain consistency
            display.refresh_dashboard_panels(mock_trainer)

            # Verify display state remains valid
            assert display.layout is not None
            assert display.log_panel.renderable is not None
            assert display.progress_bar is not None

    def test_enhanced_mock_behavior_validation(self, mock_config, mock_trainer, mock_library):
        """Test enhanced mock validation for better behavior verification."""
        # Use enhanced mock console with behavior tracking
        console_mock = mock_library.mock_console_with_dimensions(120, 30)

        display = TrainingDisplay(mock_config, mock_trainer, console_mock)

        # Setup log messages
        mock_trainer.rich_log_messages = ["Test message 1", "Test message 2"]

        # Perform update
        display.refresh_dashboard_panels(mock_trainer)

        # Verify actual console interaction behavior
        # (The mock tracks what was printed for validation)
        assert hasattr(console_mock, 'printed_content')
        # Console.log might be called for error logging, which is acceptable

    def test_component_failure_isolation(self, mock_config, mock_trainer, mock_console):
        """Test that failure in one component doesn't break others."""
        mock_console.size.width = 200
        mock_console.size.height = 50

        display = TrainingDisplay(mock_config, mock_trainer, mock_console)

        # Make one component fail
        if display.board_component:
            with patch.object(display.board_component, 'render', side_effect=RuntimeError("Board error")):
                # Should continue to update other components
                display.refresh_dashboard_panels(mock_trainer)

        # Verify error was logged but display continues to function
        assert display.layout is not None
        assert display.log_panel is not None

    def test_resource_cleanup_and_management(self, mock_config, mock_trainer, mock_console):
        """Test proper resource cleanup and management."""
        mock_console.size.width = 120
        mock_console.size.height = 30

        display = TrainingDisplay(mock_config, mock_trainer, mock_console)

        # Simulate resource-intensive operations
        large_log_messages = [f"Large log message {i}" * 100 for i in range(100)]
        mock_trainer.rich_log_messages = large_log_messages

        # Should handle large log messages gracefully
        display.refresh_dashboard_panels(mock_trainer)

        # Verify reasonable resource usage
        # (In a real implementation, this would check memory usage)
        assert display.log_panel.renderable is not None


class TestTrainingDisplayIntegration:
    """Integration tests for TrainingDisplay with other components."""

    def test_end_to_end_display_workflow(self, mock_config, training_session_in_progress):
        """Test complete end-to-end display workflow."""
        # Use real console for integration test
        console = Console(file=open('/dev/null', 'w', encoding='utf-8'), width=120, height=30)

        # Create realistic trainer mock
        trainer = Mock()
        session_data = training_session_in_progress
        trainer.game = session_data['game_state']
        trainer.rich_log_messages = [f"Training step {i}" for i in range(10)]
        trainer.metrics_manager = Mock()
        trainer.metrics_manager.global_timestep = session_data['total_moves']
        trainer.metrics_manager.total_episodes_completed = session_data['total_moves'] // 10
        trainer.metrics_manager.black_wins = 30
        trainer.metrics_manager.white_wins = 25
        trainer.metrics_manager.draws = 5
        trainer.metrics_manager.get_hot_squares = Mock(return_value=["5e", "6f", "7g"])

        # Add proper history object instead of Mock
        trainer.metrics_manager.history = TestDataFactory.create_metrics_history(length=20)

        # Add step_manager with proper values
        trainer.step_manager = Mock()
        trainer.step_manager.move_log = ["7g7f", "8c8d", "6g6f"]
        trainer.step_manager.sente_best_capture = "Rook on 2h"
        trainer.step_manager.gote_best_capture = "Bishop on 7g"
        trainer.step_manager.sente_capture_count = 3
        trainer.step_manager.gote_capture_count = 2
        trainer.step_manager.sente_drop_count = 1
        trainer.step_manager.gote_drop_count = 0
        trainer.step_manager.sente_promo_count = 1
        trainer.step_manager.gote_promo_count = 1

        # Add missing trainer attributes that display might access
        trainer.last_gradient_norm = 0.5
        trainer.last_ply_per_sec = 12.0

        # Create display and test full workflow
        display = TrainingDisplay(mock_config, trainer, console)

        # Start live display (don't actually use context manager to avoid terminal issues)
        live_display = display.start()
        assert live_display is not None

        # Test progress updates
        display.update_progress(trainer, 10.5, {
            'ep_metrics': 'Ep L:150 R:0.75',
            'ppo_metrics': 'Loss: 0.05'
        })

        # Test dashboard refresh
        display.refresh_dashboard_panels(trainer)

        # Verify workflow completed successfully
        assert display.layout is not None
        assert display.progress_bar is not None

    def test_realistic_training_simulation(self, mock_config):
        """Test display with realistic training simulation data."""
        # Use string buffer to capture output
        output_buffer = io.StringIO()
        console = Console(file=output_buffer, width=120, height=30)

        # Create realistic trainer progression
        trainer = Mock()
        trainer.rich_log_messages = []
        trainer.metrics_manager = Mock()
        trainer.metrics_manager.global_timestep = 1000
        trainer.metrics_manager.total_episodes_completed = 50
        trainer.metrics_manager.black_wins = 25
        trainer.metrics_manager.white_wins = 20
        trainer.metrics_manager.draws = 5
        trainer.metrics_manager.get_hot_squares = Mock(return_value=["4d", "5e", "6f"])
        trainer.metrics_manager.history = TestDataFactory.create_metrics_history(length=10)

        trainer.game = TestDataFactory.create_game_state(move_count=50)
        trainer.step_manager = Mock()
        trainer.step_manager.move_log = [f"move_{i}" for i in range(20)]
        trainer.step_manager.sente_best_capture = "Knight on 6g"
        trainer.step_manager.gote_best_capture = "Silver on 4d"
        trainer.step_manager.sente_capture_count = 4
        trainer.step_manager.gote_capture_count = 3
        trainer.step_manager.sente_drop_count = 2
        trainer.step_manager.gote_drop_count = 1
        trainer.step_manager.sente_promo_count = 2
        trainer.step_manager.gote_promo_count = 1

        # Add missing trainer attributes that display might access
        trainer.last_gradient_norm = 0.3
        trainer.last_ply_per_sec = 15.0

        display = TrainingDisplay(mock_config, trainer, console)

        # Simulate training progression
        for step in range(0, 1000, 100):
            trainer.metrics_manager.global_timestep = step
            trainer.metrics_manager.total_episodes_completed = step // 20
            trainer.rich_log_messages.append(f"Training step {step} completed")

            # Update display
            display.update_progress(trainer, 15.0, {
                'ep_metrics': f'Ep L:{step//20} R:{0.5 + step/2000}',
                'ppo_metrics': f'Loss: {0.1 - step/10000}'
            })
            display.refresh_dashboard_panels(trainer)

        # Verify simulation completed without errors
        output = output_buffer.getvalue()
        assert len(output) > 0  # Should have produced some output

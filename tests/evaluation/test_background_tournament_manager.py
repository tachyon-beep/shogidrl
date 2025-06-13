"""
Tests for BackgroundTournamentManager - comprehensive coverage for tournament execution.

This module provides full test coverage for the BackgroundTournamentManager class,
including tournament execution, progress tracking, error handling, and resource management.
"""

import asyncio
import json
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from keisei.evaluation.core import (
    AgentInfo,
    EvaluationContext,
    EvaluationResult,
    GameResult,
    OpponentInfo,
)
from keisei.evaluation.core.background_tournament import (
    BackgroundTournamentManager,
    TournamentProgress,
    TournamentStatus,
)
from keisei.evaluation.core.evaluation_result import SummaryStats


@pytest.fixture
def tournament_manager():
    """Create a BackgroundTournamentManager instance for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        result_dir = Path(temp_dir) / "tournament_results"
        manager = BackgroundTournamentManager(
            max_concurrent_tournaments=2,
            result_storage_dir=result_dir,
        )
        yield manager


@pytest.fixture
def mock_agent_info():
    """Create a mock AgentInfo for testing."""
    return AgentInfo(
        name="TestAgent",
        checkpoint_path="/tmp/test_agent.pt",
        model_type="PPO",
        training_timesteps=100000,
        version="1.0",
        metadata={"elo_rating": 1200.0, "games_played": 10, "win_rate": 0.75},
    )


@pytest.fixture
def mock_opponents():
    """Create mock OpponentInfo instances for testing."""
    return [
        OpponentInfo(
            name="Opponent1",
            type="ppo",
            checkpoint_path="/tmp/opponent1.pt",
            difficulty_level=1100.0,
            version="1.0",
            metadata={"elo_rating": 1100.0, "games_played": 20, "win_rate": 0.60},
        ),
        OpponentInfo(
            name="Opponent2",
            type="ppo",
            checkpoint_path="/tmp/opponent2.pt",
            difficulty_level=1300.0,
            version="1.0",
            metadata={"elo_rating": 1300.0, "games_played": 15, "win_rate": 0.80},
        ),
    ]


@pytest.fixture
def mock_tournament_config():
    """Create a mock tournament configuration."""
    config = Mock()
    config.num_games_per_opponent = 2
    config.max_moves_per_game = 100
    config.timeout_per_game = 60.0
    return config


class TestBackgroundTournamentManager:
    """Test suite for BackgroundTournamentManager functionality."""

    def test_initialization(self):
        """Test BackgroundTournamentManager initialization."""
        progress_callback = Mock()

        with tempfile.TemporaryDirectory() as temp_dir:
            result_dir = Path(temp_dir)
            manager = BackgroundTournamentManager(
                max_concurrent_tournaments=3,
                progress_callback=progress_callback,
                result_storage_dir=result_dir,
            )

            assert manager.max_concurrent_tournaments == 3
            assert manager.progress_callback == progress_callback
            assert manager.result_storage_dir == result_dir
            assert result_dir.exists()
            assert len(manager._active_tournaments) == 0
            assert len(manager._tournament_tasks) == 0

    def test_initialization_default_storage_dir(self):
        """Test initialization with default storage directory."""
        manager = BackgroundTournamentManager()

        assert manager.result_storage_dir == Path("./tournament_results")
        assert manager.max_concurrent_tournaments == 2
        assert manager.progress_callback is None

    @pytest.mark.asyncio
    async def test_start_tournament_success(
        self,
        tournament_manager,
        mock_agent_info,
        mock_opponents,
        mock_tournament_config,
    ):
        """Test successful tournament start."""
        with patch(
            "keisei.evaluation.core.background_tournament.TournamentEvaluator"
        ) as mock_evaluator_class:
            # Mock the evaluator instance
            mock_evaluator = AsyncMock()
            mock_evaluator_class.return_value = mock_evaluator

            # Mock the _execute_with_progress_tracking method
            mock_result = EvaluationResult(
                context=Mock(),
                games=[],
                summary_stats=Mock(),
                analytics_data={"tournament_specific_analytics": {}},
            )
            mock_evaluator._execute_with_progress_tracking = AsyncMock(
                return_value=mock_result
            )
            mock_evaluator._play_games_against_opponent = AsyncMock(
                return_value=([], [])
            )
            mock_evaluator._calculate_tournament_standings = Mock(return_value={})

            tournament_id = await tournament_manager.start_tournament(
                tournament_config=mock_tournament_config,
                agent_info=mock_agent_info,
                opponents=mock_opponents,
                tournament_name="test_tournament",
            )

            assert tournament_id is not None
            assert tournament_id.startswith("test_tournament_")
            assert len(tournament_manager._active_tournaments) == 1
            assert len(tournament_manager._tournament_tasks) == 1

            # Verify tournament progress was created
            progress = tournament_manager.get_tournament_progress(tournament_id)
            assert progress is not None
            assert progress.tournament_id == tournament_id
            assert progress.total_games == 4  # 2 opponents * 2 games each
            assert progress.status == TournamentStatus.CREATED

    @pytest.mark.asyncio
    async def test_start_tournament_without_name(
        self,
        tournament_manager,
        mock_agent_info,
        mock_opponents,
        mock_tournament_config,
    ):
        """Test tournament start without custom name."""
        with patch("keisei.evaluation.core.background_tournament.TournamentEvaluator"):
            tournament_id = await tournament_manager.start_tournament(
                tournament_config=mock_tournament_config,
                agent_info=mock_agent_info,
                opponents=mock_opponents,
            )

            assert tournament_id is not None
            # Should be a UUID format without custom prefix
            try:
                uuid.UUID(tournament_id)
                is_uuid = True
            except ValueError:
                is_uuid = False
            assert is_uuid

    @pytest.mark.asyncio
    async def test_tournament_execution_progress_tracking(
        self,
        tournament_manager,
        mock_agent_info,
        mock_opponents,
        mock_tournament_config,
    ):
        """Test tournament execution with progress tracking."""
        progress_updates = []

        def progress_callback(progress):
            progress_updates.append((progress.status, progress.completion_percentage))

        tournament_manager.progress_callback = progress_callback

        with patch(
            "keisei.evaluation.core.background_tournament.TournamentEvaluator"
        ) as mock_evaluator_class:
            mock_evaluator = AsyncMock()
            mock_evaluator_class.return_value = mock_evaluator

            # Mock game execution
            mock_games = [Mock(), Mock()]  # 2 games
            mock_evaluator._play_games_against_opponent = AsyncMock(
                return_value=(mock_games, [])
            )
            mock_evaluator._calculate_tournament_standings = Mock(
                return_value={"standings": "test"}
            )

            # Start tournament
            tournament_id = await tournament_manager.start_tournament(
                tournament_config=mock_tournament_config,
                agent_info=mock_agent_info,
                opponents=mock_opponents[:1],  # Use only 1 opponent for simpler testing
            )

            # Wait for tournament to complete
            await asyncio.sleep(0.5)  # Allow tournament to run

            # Check that progress was tracked
            final_progress = tournament_manager.get_tournament_progress(tournament_id)
            assert final_progress is not None

            # Verify progress callbacks were called
            assert len(progress_updates) > 0
            assert any(
                status == TournamentStatus.RUNNING for status, _ in progress_updates
            )

    @pytest.mark.asyncio
    async def test_tournament_error_handling(
        self,
        tournament_manager,
        mock_agent_info,
        mock_opponents,
        mock_tournament_config,
    ):
        """Test tournament error handling."""
        with patch(
            "keisei.evaluation.core.background_tournament.TournamentEvaluator"
        ) as mock_evaluator_class:
            # Mock evaluator to raise an exception
            mock_evaluator_class.side_effect = Exception("Tournament execution failed")

            tournament_id = await tournament_manager.start_tournament(
                tournament_config=mock_tournament_config,
                agent_info=mock_agent_info,
                opponents=mock_opponents,
            )

            # Wait for tournament to fail
            await asyncio.sleep(0.5)

            # Check that tournament failed
            progress = tournament_manager.get_tournament_progress(tournament_id)
            assert progress.status == TournamentStatus.FAILED
            assert "Tournament execution failed" in progress.error_message

    @pytest.mark.asyncio
    async def test_cancel_tournament(
        self,
        tournament_manager,
        mock_agent_info,
        mock_opponents,
        mock_tournament_config,
    ):
        """Test tournament cancellation."""
        with patch(
            "keisei.evaluation.core.background_tournament.TournamentEvaluator"
        ) as mock_evaluator_class:
            mock_evaluator = AsyncMock()
            mock_evaluator_class.return_value = mock_evaluator

            # Make tournament execution take a while
            async def slow_execution(*args, **kwargs):
                await asyncio.sleep(2.0)
                return Mock()

            mock_evaluator._execute_with_progress_tracking = slow_execution

            tournament_id = await tournament_manager.start_tournament(
                tournament_config=mock_tournament_config,
                agent_info=mock_agent_info,
                opponents=mock_opponents,
            )

            # Let tournament start
            await asyncio.sleep(0.1)

            # Cancel tournament
            result = await tournament_manager.cancel_tournament(tournament_id)
            assert result is True

            # Check status
            progress = tournament_manager.get_tournament_progress(tournament_id)
            assert progress.status == TournamentStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_cancel_nonexistent_tournament(self, tournament_manager):
        """Test cancelling a tournament that doesn't exist."""
        result = await tournament_manager.cancel_tournament("nonexistent_id")
        assert result is False

    def test_list_active_tournaments(
        self,
        tournament_manager,
        mock_agent_info,
        mock_opponents,
        mock_tournament_config,
    ):
        """Test listing active tournaments."""
        # Initially no tournaments
        active = tournament_manager.list_active_tournaments()
        assert len(active) == 0

        # Add some tournament progress manually for testing
        progress1 = TournamentProgress(
            tournament_id="test1",
            status=TournamentStatus.RUNNING,
            total_games=10,
        )
        progress2 = TournamentProgress(
            tournament_id="test2",
            status=TournamentStatus.COMPLETED,
            total_games=5,
        )
        progress3 = TournamentProgress(
            tournament_id="test3",
            status=TournamentStatus.PAUSED,
            total_games=8,
        )

        tournament_manager._active_tournaments = {
            "test1": progress1,
            "test2": progress2,
            "test3": progress3,
        }

        active = tournament_manager.list_active_tournaments()
        assert len(active) == 2  # Only RUNNING and PAUSED
        assert any(t.tournament_id == "test1" for t in active)
        assert any(t.tournament_id == "test3" for t in active)
        assert not any(t.tournament_id == "test2" for t in active)

    def test_list_all_tournaments(self, tournament_manager):
        """Test listing all tournaments."""
        # Add some tournament progress
        progress1 = TournamentProgress(
            tournament_id="test1", status=TournamentStatus.RUNNING
        )
        progress2 = TournamentProgress(
            tournament_id="test2", status=TournamentStatus.COMPLETED
        )

        tournament_manager._active_tournaments = {
            "test1": progress1,
            "test2": progress2,
        }

        all_tournaments = tournament_manager.list_all_tournaments()
        assert len(all_tournaments) == 2
        assert any(t.tournament_id == "test1" for t in all_tournaments)
        assert any(t.tournament_id == "test2" for t in all_tournaments)

    def test_get_tournament_progress(self, tournament_manager):
        """Test getting tournament progress."""
        progress = TournamentProgress(
            tournament_id="test123", status=TournamentStatus.RUNNING
        )
        tournament_manager._active_tournaments["test123"] = progress

        retrieved = tournament_manager.get_tournament_progress("test123")
        assert retrieved == progress

        # Test nonexistent tournament
        none_retrieved = tournament_manager.get_tournament_progress("nonexistent")
        assert none_retrieved is None

    @pytest.mark.asyncio
    async def test_shutdown(
        self,
        tournament_manager,
        mock_agent_info,
        mock_opponents,
        mock_tournament_config,
    ):
        """Test manager shutdown with running tournaments."""
        with patch(
            "keisei.evaluation.core.background_tournament.TournamentEvaluator"
        ) as mock_evaluator_class:
            mock_evaluator = AsyncMock()
            mock_evaluator_class.return_value = mock_evaluator

            # Make tournament execution take a while
            async def slow_execution(*args, **kwargs):
                await asyncio.sleep(1.0)
                return Mock()

            mock_evaluator._execute_with_progress_tracking = slow_execution

            # Start a tournament
            tournament_id = await tournament_manager.start_tournament(
                tournament_config=mock_tournament_config,
                agent_info=mock_agent_info,
                opponents=mock_opponents,
            )

            # Let tournament start
            await asyncio.sleep(0.1)

            # Shutdown manager
            await tournament_manager.shutdown()

            # Check that tournament was cancelled
            progress = tournament_manager.get_tournament_progress(tournament_id)
            assert progress.status == TournamentStatus.CANCELLED

            # Check that shutdown event was set
            assert tournament_manager._shutdown_event.is_set()

    @pytest.mark.asyncio
    async def test_result_saving(self, tournament_manager):
        """Test tournament result saving functionality."""
        tournament_id = "test_save_results"

        # Create a mock result with proper game result mocks
        mock_games = [Mock(spec=GameResult)]
        mock_games[0].moves_count = 50
        mock_games[0].duration_seconds = 30.0
        mock_games[0].is_agent_win = True
        mock_games[0].is_opponent_win = False
        mock_games[0].is_draw = False

        # Create summary stats using from_games method
        summary_stats = SummaryStats.from_games(mock_games)

        mock_context = Mock()
        mock_context.session_id = tournament_id
        mock_context.to_dict = Mock(
            return_value={"session_id": tournament_id, "test_context": "data"}
        )

        mock_result = EvaluationResult(
            context=mock_context,
            games=mock_games,
            summary_stats=summary_stats,
            analytics_data={"test": "data"},
        )

        await tournament_manager._save_tournament_results(tournament_id, mock_result)

        # Check that result file was created
        result_file = (
            tournament_manager.result_storage_dir / f"{tournament_id}_results.json"
        )
        assert result_file.exists()

        # Verify content matches our implementation's structure
        with open(result_file, "r") as f:
            saved_data = json.load(f)

        # Verify the structure our implementation actually saves
        assert saved_data["tournament_id"] == tournament_id
        assert "context" in saved_data
        assert "timestamp" in saved_data
        assert "summary_stats" in saved_data
        assert "analytics_data" in saved_data
        assert saved_data["analytics_data"] == {"test": "data"}
        assert saved_data["total_games"] == 1

    def test_tournament_progress_properties(self):
        """Test TournamentProgress property methods."""
        progress = TournamentProgress(
            tournament_id="test",
            status=TournamentStatus.RUNNING,
            total_games=10,
            completed_games=3,
        )

        # Test completion percentage
        assert progress.completion_percentage == 30.0

        # Test is_active
        assert progress.is_active is True

        # Test is_complete
        assert progress.is_complete is False

        # Test with completed status
        progress.status = TournamentStatus.COMPLETED
        assert progress.is_active is False
        assert progress.is_complete is True

        # Test with zero total games
        progress.total_games = 0
        assert progress.completion_percentage == 0.0

    @pytest.mark.asyncio
    async def test_concurrent_tournament_limit(
        self, mock_agent_info, mock_opponents, mock_tournament_config
    ):
        """Test that concurrent tournament limit is enforced."""
        # Create manager with limit of 1
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = BackgroundTournamentManager(
                max_concurrent_tournaments=1,
                result_storage_dir=Path(temp_dir),
            )

            with patch(
                "keisei.evaluation.core.background_tournament.TournamentEvaluator"
            ) as mock_evaluator_class:
                mock_evaluator = AsyncMock()
                mock_evaluator_class.return_value = mock_evaluator

                # Make tournaments take time
                async def slow_execution(*args, **kwargs):
                    await asyncio.sleep(0.5)
                    return Mock()

                mock_evaluator._execute_with_progress_tracking = slow_execution

                # Start first tournament
                tournament_id1 = await manager.start_tournament(
                    tournament_config=mock_tournament_config,
                    agent_info=mock_agent_info,
                    opponents=mock_opponents[:1],
                )

                # Start second tournament (should wait due to semaphore)
                tournament_id2 = await manager.start_tournament(
                    tournament_config=mock_tournament_config,
                    agent_info=mock_agent_info,
                    opponents=mock_opponents[:1],
                )

                # Both should be created but only one running at first
                assert tournament_id1 != tournament_id2
                assert len(manager._active_tournaments) == 2
                assert len(manager._tournament_tasks) == 2

    @pytest.mark.asyncio
    async def test_tournament_cleanup(self, tournament_manager):
        """Test tournament cleanup functionality."""
        tournament_id = "test_cleanup"

        # Add mock task and lock
        mock_task = Mock()
        mock_task.done.return_value = False
        tournament_manager._tournament_tasks[tournament_id] = mock_task
        tournament_manager._tournament_locks[tournament_id] = asyncio.Lock()

        # Cleanup
        await tournament_manager._cleanup_tournament(tournament_id)

        # Verify cleanup
        assert tournament_id not in tournament_manager._tournament_tasks
        assert tournament_id not in tournament_manager._tournament_locks
        mock_task.cancel.assert_called_once()


class TestTournamentProgressIntegration:
    """Integration tests for TournamentProgress tracking."""

    def test_progress_performance_metrics(self):
        """Test progress performance metric calculations."""
        progress = TournamentProgress(
            tournament_id="test",
            total_games=100,
            completed_games=50,
            games_per_second=2.5,
            average_game_duration=0.4,
        )

        assert progress.completion_percentage == 50.0
        assert abs(progress.games_per_second - 2.5) < 0.001
        assert abs(progress.average_game_duration - 0.4) < 0.001

    def test_progress_with_results(self):
        """Test progress tracking with game results."""
        from keisei.evaluation.core import GameResult

        mock_results = [
            Mock(spec=GameResult),
            Mock(spec=GameResult),
        ]

        progress = TournamentProgress(
            tournament_id="test",
            total_games=10,
            completed_games=2,
            results=mock_results,
        )

        assert len(progress.results) == 2
        assert progress.completion_percentage == 20.0

    def test_progress_with_standings(self):
        """Test progress tracking with tournament standings."""
        standings = {
            "TestAgent": {"wins": 5, "losses": 2, "elo": 1250},
            "Opponent1": {"wins": 3, "losses": 4, "elo": 1180},
        }

        progress = TournamentProgress(
            tournament_id="test",
            standings=standings,
        )

        assert progress.standings == standings
        assert "TestAgent" in progress.standings
        assert progress.standings["TestAgent"]["wins"] == 5

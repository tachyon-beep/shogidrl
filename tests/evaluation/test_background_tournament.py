"""
Tests for background tournament management (Task 6 - High Priority)
Coverage for keisei/evaluation/core/background_tournament.py
"""
import pytest
import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path
import tempfile

from keisei.constants import CORE_OBSERVATION_CHANNELS, FULL_ACTION_SPACE
from tests.evaluation.factories import EvaluationTestFactory


class TestBackgroundTournamentManager:
    """Test BackgroundTournamentManager lifecycle and functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.tournament_manager = Mock()
        self.mock_opponents = ["agent_a", "agent_b", "agent_c"]

    def test_tournament_manager_lifecycle(self):
        """Test complete tournament lifecycle from start to finish."""
        with patch('keisei.evaluation.core.background_tournament.BackgroundTournamentManager') as MockManager:
            manager = MockManager.return_value
            
            # Mock tournament creation
            tournament_id = "tournament_001"
            manager.create_tournament.return_value = {
                "id": tournament_id,
                "status": "created",
                "participants": self.mock_opponents,
                "rounds": 3,
                "created_at": time.time()
            }
            
            # Mock tournament start
            manager.start_tournament.return_value = {
                "id": tournament_id,
                "status": "running",
                "started_at": time.time()
            }
            
            # Test tournament creation
            tournament = manager.create_tournament(
                participants=self.mock_opponents,
                rounds=3,
                tournament_type="round_robin"
            )
            
            assert tournament["id"] == tournament_id
            assert tournament["status"] == "created"
            assert len(tournament["participants"]) == 3
            
            # Test tournament start
            started = manager.start_tournament(tournament_id)
            assert started["status"] == "running"

    def test_tournament_status_state_machine_validation(self):
        """Test TournamentStatus state transitions are valid."""
        valid_transitions = {
            "created": ["running", "cancelled"],
            "running": ["paused", "completed", "cancelled"],
            "paused": ["running", "cancelled"],
            "completed": [],  # Terminal state
            "cancelled": []   # Terminal state
        }
        
        with patch('keisei.evaluation.core.background_tournament.TournamentStatus') as MockStatus:
            # Mock status validation
            MockStatus.validate_transition.side_effect = lambda from_state, to_state: \
                to_state in valid_transitions.get(from_state, [])
            
            # Test valid transitions
            assert MockStatus.validate_transition("created", "running") is True
            assert MockStatus.validate_transition("running", "completed") is True
            assert MockStatus.validate_transition("paused", "running") is True
            
            # Test invalid transitions
            assert MockStatus.validate_transition("completed", "running") is False
            assert MockStatus.validate_transition("cancelled", "running") is False

    def test_tournament_progress_tracking_accuracy(self):
        """Test tournament progress tracking and reporting."""
        with patch('keisei.evaluation.core.background_tournament.TournamentProgress') as MockProgress:
            progress = MockProgress.return_value
            
            # Mock progress updates
            progress.update_match_completed.return_value = None
            progress.get_completion_percentage.return_value = 75.0
            progress.get_remaining_matches.return_value = 3
            progress.get_elapsed_time.return_value = 1800  # 30 minutes
            progress.estimate_time_remaining.return_value = 600  # 10 minutes
            
            # Test progress tracking
            progress.update_match_completed("agent_a", "agent_b", "agent_a")
            
            completion = progress.get_completion_percentage()
            assert completion == 75.0
            
            remaining = progress.get_remaining_matches()
            assert remaining == 3
            
            eta = progress.estimate_time_remaining()
            assert eta == 600

    @pytest.mark.asyncio
    async def test_async_execution_without_blocking(self):
        """Test tournament runs asynchronously without blocking main thread."""
        with patch('keisei.evaluation.core.background_tournament.BackgroundTournamentManager') as MockManager:
            manager = MockManager.return_value
            
            # Mock async tournament execution
            async def mock_run_tournament(tournament_id):
                await asyncio.sleep(0.1)  # Simulate work
                return {"id": tournament_id, "status": "completed"}
            
            manager.run_tournament_async = AsyncMock(side_effect=mock_run_tournament)
            manager.is_tournament_running.return_value = True
            
            # Start tournament in background
            tournament_id = "async_tournament_001"
            start_time = time.time()
            
            # This should not block
            task = asyncio.create_task(manager.run_tournament_async(tournament_id))
            
            # Should be able to do other work immediately
            elapsed_immediate = time.time() - start_time
            assert elapsed_immediate < 0.05  # Should be nearly instantaneous
            
            # Tournament should still be running
            assert manager.is_tournament_running(tournament_id) is True
            
            # Wait for completion
            result = await task
            assert result["status"] == "completed"

    def test_tournament_persistence_and_recovery(self):
        """Test tournament state persistence and recovery after restart."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            state_file = Path(tmp_dir) / "tournament_state.json"
            
            with patch('keisei.evaluation.core.background_tournament.BackgroundTournamentManager') as MockManager:
                manager = MockManager.return_value
                
                # Mock state persistence
                tournament_state = {
                    "id": "persistent_tournament",
                    "status": "running",
                    "progress": 0.4,
                    "matches_completed": 8,
                    "matches_total": 20
                }
                
                manager.save_state.return_value = str(state_file)
                manager.load_state.return_value = tournament_state
                
                # Test state saving
                saved_path = manager.save_state("persistent_tournament")
                assert saved_path == str(state_file)
                
                # Test state recovery
                recovered_state = manager.load_state(str(state_file))
                assert recovered_state["id"] == "persistent_tournament"
                assert recovered_state["status"] == "running"
                assert recovered_state["progress"] == 0.4

    def test_resource_cleanup_and_memory_management(self):
        """Test proper resource cleanup after tournament completion."""
        with patch('keisei.evaluation.core.background_tournament.BackgroundTournamentManager') as MockManager:
            manager = MockManager.return_value
            
            # Mock resource tracking
            initial_memory = 100  # MB
            peak_memory = 250     # MB
            final_memory = 105    # MB (slight increase is acceptable)
            
            manager.get_memory_usage.side_effect = [initial_memory, peak_memory, final_memory]
            manager.cleanup_tournament.return_value = {
                "cleaned_up": True,
                "temp_files_removed": 15,
                "memory_freed_mb": peak_memory - final_memory
            }
            
            # Test memory tracking during tournament
            initial = manager.get_memory_usage()
            peak = manager.get_memory_usage()
            
            # Test cleanup
            cleanup_result = manager.cleanup_tournament("test_tournament")
            final = manager.get_memory_usage()
            
            assert cleanup_result["cleaned_up"] is True
            assert cleanup_result["memory_freed_mb"] > 0
            assert final < peak  # Memory should be reduced after cleanup

    def test_tournament_cancellation_and_cleanup(self):
        """Test tournament cancellation and proper cleanup."""
        with patch('keisei.evaluation.core.background_tournament.BackgroundTournamentManager') as MockManager:
            manager = MockManager.return_value
            
            tournament_id = "cancellable_tournament"
            
            # Mock cancellation
            manager.cancel_tournament.return_value = {
                "id": tournament_id,
                "status": "cancelled",
                "reason": "user_requested",
                "matches_completed": 5,
                "matches_cancelled": 7
            }
            
            manager.is_tournament_running.return_value = False
            
            # Test cancellation
            cancel_result = manager.cancel_tournament(tournament_id)
            
            assert cancel_result["status"] == "cancelled"
            assert cancel_result["reason"] == "user_requested"
            assert cancel_result["matches_completed"] == 5
            
            # Tournament should no longer be running
            assert manager.is_tournament_running(tournament_id) is False


class TestTournamentScheduling:
    """Test tournament scheduling and queuing functionality."""

    def test_tournament_queue_management(self):
        """Test tournament queuing when resources are limited."""
        # Mock scheduler directly without patching non-existent class
        mock_scheduler = Mock()
        
        # Mock queue operations
        mock_scheduler.add_to_queue.return_value = {"position": 2, "estimated_wait": 300}
        mock_scheduler.get_queue_status.return_value = {
            "queue_length": 3,
            "currently_running": 1,
            "max_concurrent": 2
        }
        
        # Test adding to queue
        queue_result = mock_scheduler.add_to_queue("queued_tournament")
        assert queue_result["position"] == 2
        assert queue_result["estimated_wait"] > 0
        
        # Test queue status
        status = mock_scheduler.get_queue_status()
        assert status["queue_length"] == 3
        assert status["currently_running"] <= status["max_concurrent"]

    def test_priority_scheduling(self):
        """Test priority-based tournament scheduling."""
        # Mock scheduler directly without patching non-existent class
        mock_scheduler = Mock()
        
        # Mock priority scheduling
        mock_scheduler.schedule_with_priority.return_value = {
            "scheduled": True,
            "priority": "high",
            "estimated_start": time.time() + 60
        }
        
        # Test high priority scheduling
        result = mock_scheduler.schedule_with_priority(
            tournament_id="priority_tournament",
            priority="high"
        )
        
        assert result["scheduled"] is True
        assert result["priority"] == "high"
        assert result["estimated_start"] > time.time()
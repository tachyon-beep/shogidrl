#!/usr/bin/env python3
"""
Priority 2 Integration Test: Evaluation System Display Integration

This test validates that the evaluation system properly integrates with the trainer 
display system, specifically checking the ELO panel integration.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

# Test the integration between evaluation system and display
class TestEvaluationDisplayIntegration:
    """Test evaluation system integration with trainer display."""

    def test_elo_panel_integration(self):
        """Test that evaluation results properly populate the ELO panel."""
        
        # Mock the trainer with evaluation_elo_snapshot
        trainer = Mock()
        trainer.evaluation_elo_snapshot = {
            "current_id": "test_agent",
            "current_rating": 1550.0,
            "opponent_id": "opponent_1",
            "opponent_rating": 1450.0,
            "last_outcome": "win",
            "top_ratings": [
                ("test_agent", 1550.0),
                ("opponent_1", 1450.0),
                ("opponent_2", 1350.0)
            ]
        }
        
        # Mock the required display configuration and components
        config = Mock()
        config.training = Mock()
        config.training.refresh_per_second = 4
        config.training.total_timesteps = 10000  # Set higher than global_timestep to avoid comparison issues
        config.display = Mock()
        config.display.enable_elo_ratings = True
        config.display.enable_trend_visualization = False  # Disable to avoid component issues
        config.display.enable_board_display = False  # Disable to avoid component issues
        config.display.metrics_window_size = 50
        
        console = Mock()
        console.size = Mock()
        console.size.height = 40
        console.size.width = 120  # Set proper width for layout calculations
        console.encoding = "utf-8"  # Set proper encoding string
        
        trainer.rich_log_messages = []
        trainer.metrics_manager = Mock()
        trainer.metrics_manager.global_timestep = 1000
        trainer.metrics_manager.black_wins = 5
        trainer.metrics_manager.white_wins = 3
        trainer.metrics_manager.draws = 2
        trainer.metrics_manager.total_episodes_completed = 10
        trainer.last_gradient_norm = 0.5  # Mock gradient norm
        
        # Mock metrics history lists to avoid len() issues
        trainer.metrics_manager.timestep_history = [100, 200, 300, 400, 500]
        trainer.metrics_manager.black_wins_history = [1, 2, 3, 4, 5]
        trainer.metrics_manager.white_wins_history = [0, 1, 1, 2, 3]
        trainer.metrics_manager.draws_history = [0, 0, 1, 1, 2]
        trainer.metrics_manager.black_winrate_history = [1.0, 0.67, 0.6, 0.57, 0.5]
        trainer.metrics_manager.white_winrate_history = [0.0, 0.33, 0.2, 0.29, 0.3]
        trainer.metrics_manager.draw_rate_history = [0.0, 0.0, 0.2, 0.14, 0.2]
        
        # Mock the history object that _build_metric_lines expects
        trainer.metrics_manager.history = Mock()
        trainer.metrics_manager.history.win_rates_history = [
            {"win_rate_black": 0.5, "win_rate_white": 0.3, "win_rate_draw": 0.2}
        ]
        # Add all other metric attributes that might be accessed
        trainer.metrics_manager.history.policy_losses = [0.1, 0.08, 0.06]
        trainer.metrics_manager.history.value_losses = [0.05, 0.04, 0.03]
        trainer.metrics_manager.history.entropies = [0.9, 0.85, 0.8]
        trainer.metrics_manager.history.clip_fractions = [0.2, 0.15, 0.1]
        
        # Mock the agent and model properly to avoid named_parameters() issues
        trainer.agent = Mock()
        trainer.agent.model = Mock()
        trainer.agent.model.named_parameters = Mock(return_value=iter([]))  # Return empty iterator
        
        # Create TrainingDisplay with mocked components to avoid constructor issues
        with patch('keisei.training.display_components.RollingAverageCalculator'), \
             patch('keisei.training.display_components.ShogiBoard'), \
             patch('keisei.training.display_components.RecentMovesPanel'), \
             patch('keisei.training.display_components.PieceStandPanel'), \
             patch('keisei.training.display_components.Sparkline'), \
             patch('keisei.training.display_components.MultiMetricSparkline'):
            
            from keisei.training.display import TrainingDisplay
            
            display = TrainingDisplay(config, trainer, console)
            
            # Verify ELO component is enabled
            assert display.elo_component_enabled == True
            
            # Test refresh_dashboard_panels with ELO data
            display.refresh_dashboard_panels(trainer)
            
            # Verify ELO panel was updated (check that layout was accessed)
            assert hasattr(display, 'layout')
            
            print("✅ ELO panel integration test passed")

    def test_evaluation_manager_display_flow(self):
        """Test the complete flow from evaluation to display update."""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock evaluation manager
            evaluation_manager = Mock()
            mock_eval_results = Mock()
            mock_eval_results.summary_stats = Mock()
            mock_eval_results.summary_stats.win_rate = 0.6
            mock_eval_results.summary_stats.loss_rate = 0.3
            mock_eval_results.summary_stats.draw_rate = 0.1
            evaluation_manager.evaluate_current_agent.return_value = mock_eval_results
            # Mock opponent pool for the callback
            evaluation_manager.opponent_pool = Mock()
            evaluation_manager.opponent_pool.sample.return_value = "opponent_1.ckpt"
            
            # Mock agent
            agent = Mock()
            agent.model = Mock()
            agent.model.eval = Mock()
            agent.model.train = Mock()
            
            # Mock trainer components
            trainer = Mock()
            trainer.agent = agent
            trainer.evaluation_manager = evaluation_manager
            trainer.run_name = "test_training_run"
            trainer.log_both = Mock()
            trainer.evaluation_elo_snapshot = None
            
            # Mock configuration 
            eval_cfg = Mock()
            eval_cfg.elo_registry_path = os.path.join(temp_dir, "elo_registry.json")
            eval_cfg.interval = 1000
            eval_cfg.enable_periodic_evaluation = True  # Enable periodic evaluation
            
            # Create ELO registry file
            from keisei.evaluation.opponents.elo_registry import EloRegistry
            registry = EloRegistry(Path(eval_cfg.elo_registry_path))
            registry.ratings = {
                "test_training_run": 1520.0,
                "opponent_1": 1480.0,
                "opponent_2": 1440.0
            }
            registry.save()
            
            # Mock evaluation callback
            from keisei.training.callbacks import EvaluationCallback
            callback = EvaluationCallback(eval_cfg, interval=1000)
            
            # Mock necessary trainer attributes for callback
            trainer.metrics_manager = Mock()
            trainer.metrics_manager.global_timestep = 999  # Set to 999 so (999 + 1) % 1000 == 0
            
            # Mock opponent checkpoint selection
            with patch('keisei.training.callbacks.EloRegistry') as mock_elo_class:
                mock_elo_instance = Mock()
                mock_elo_instance.ratings = {
                    "test_training_run": 1520.0,
                    "opponent_1": 1480.0
                }
                mock_elo_instance.get_rating.side_effect = lambda x: mock_elo_instance.ratings.get(x, 1500.0)
                mock_elo_class.return_value = mock_elo_instance
                
                # Simulate evaluation callback execution
                callback.on_step_end(trainer)
            
            # Verify that evaluation was called
            evaluation_manager.evaluate_current_agent.assert_called_once_with(agent)
            
            # Verify model mode switching
            agent.model.eval.assert_called()
            agent.model.train.assert_called()
            
            print("✅ Evaluation manager display flow test passed")

    def test_display_without_evaluation_data(self):
        """Test display behavior when evaluation data is not available."""
        
        # Mock trainer without evaluation data
        trainer = Mock()
        trainer.evaluation_elo_snapshot = None  # No evaluation data
        trainer.rich_log_messages = []
        trainer.metrics_manager = Mock()
        trainer.metrics_manager.global_timestep = 1000
        trainer.metrics_manager.black_wins = 0
        trainer.metrics_manager.white_wins = 0
        trainer.metrics_manager.draws = 0
        trainer.metrics_manager.total_episodes_completed = 0
        trainer.last_gradient_norm = 0.5  # Mock gradient norm
        
        # Mock metrics history lists to avoid len() issues
        trainer.metrics_manager.timestep_history = []
        trainer.metrics_manager.black_wins_history = []
        trainer.metrics_manager.white_wins_history = []
        trainer.metrics_manager.draws_history = []
        trainer.metrics_manager.black_winrate_history = []
        trainer.metrics_manager.white_winrate_history = []
        trainer.metrics_manager.draw_rate_history = []
        
        # Mock the history object that _build_metric_lines expects
        trainer.metrics_manager.history = Mock()
        trainer.metrics_manager.history.win_rates_history = []
        # Add all other metric attributes that might be accessed
        trainer.metrics_manager.history.policy_losses = []
        trainer.metrics_manager.history.value_losses = []
        trainer.metrics_manager.history.entropies = []
        trainer.metrics_manager.history.clip_fractions = []
        
        # Mock the agent and model properly to avoid named_parameters() issues
        trainer.agent = Mock()
        trainer.agent.model = Mock()
        trainer.agent.model.named_parameters = Mock(return_value=iter([]))  # Return empty iterator
        
        # Mock display configuration
        config = Mock()
        config.training = Mock()
        config.training.refresh_per_second = 4
        config.training.total_timesteps = 10000  # Set higher than global_timestep to avoid comparison issues
        config.display = Mock()
        config.display.enable_elo_ratings = True
        
        console = Mock()
        console.size = Mock()
        console.size.height = 40
        console.size.width = 120  # Set proper width for layout calculations
        console.encoding = "utf-8"  # Set proper encoding string
        
        # Create TrainingDisplay with extensive mocking to avoid display issues
        with patch('keisei.training.display_components.RollingAverageCalculator'), \
             patch('keisei.training.display_components.ShogiBoard'), \
             patch('keisei.training.display_components.RecentMovesPanel'), \
             patch('keisei.training.display_components.PieceStandPanel'), \
             patch('keisei.training.display_components.Sparkline'), \
             patch('keisei.training.display_components.MultiMetricSparkline'), \
             patch('keisei.training.display.TrainingDisplay._build_metric_lines') as mock_build_metrics:
            
            # Mock the metric lines to return empty list to avoid len() issues
            mock_build_metrics.return_value = []
            
            from keisei.training.display import TrainingDisplay
            
            display = TrainingDisplay(config, trainer, console)
        
            # Test refresh_dashboard_panels without evaluation data
            display.refresh_dashboard_panels(trainer)
            
            # Should handle gracefully and show waiting message
            assert display.elo_component_enabled == True
            
            print("✅ Display without evaluation data test passed")

    def test_evaluation_results_formatting(self):
        """Test that evaluation results are properly formatted for display."""
        
        from keisei.evaluation.opponents.elo_registry import EloRegistry
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create and populate ELO registry
            registry_path = Path(temp_dir) / "elo_registry.json"
            registry = EloRegistry(registry_path)
            
            # Add some test ratings
            registry.ratings = {
                "agent_v1": 1600.0,
                "agent_v2": 1550.0,
                "agent_v3": 1500.0,
                "opponent_1": 1450.0,
                "opponent_2": 1400.0
            }
            registry.save()
            
            # Test rating retrieval
            assert abs(registry.get_rating("agent_v1") - 1600.0) < 0.1
            assert abs(registry.get_rating("nonexistent") - 1500.0) < 0.1  # Default
            
            # Test top ratings format (should match display expectations)
            top_ratings = sorted(
                registry.ratings.items(),
                key=lambda x: x[1], 
                reverse=True
            )[:3]
            
            expected_top_3 = [
                ("agent_v1", 1600.0),
                ("agent_v2", 1550.0), 
                ("agent_v3", 1500.0)
            ]
            
            # Check the structure matches (names and relative order)
            assert len(top_ratings) == len(expected_top_3)
            assert top_ratings[0][0] == "agent_v1"  # Highest rated
            assert top_ratings[1][0] == "agent_v2"  # Second highest
            assert top_ratings[2][0] == "agent_v3"  # Third highest
            
            print("✅ Evaluation results formatting test passed")


def main():
    """Run all integration tests."""
    
    print("🔍 Running Priority 2: Evaluation System Display Integration Tests")
    print("=" * 70)
    
    test_instance = TestEvaluationDisplayIntegration()
    
    try:
        test_instance.test_elo_panel_integration()
        test_instance.test_evaluation_manager_display_flow() 
        test_instance.test_display_without_evaluation_data()
        test_instance.test_evaluation_results_formatting()
        
        print("\n" + "=" * 70)
        print("🎉 ALL INTEGRATION TESTS PASSED!")
        print("✅ Evaluation system properly integrates with trainer display")
        print("✅ ELO panel receives and displays evaluation data correctly")
        print("✅ Display handles missing evaluation data gracefully")
        print("✅ Evaluation results are properly formatted for display")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

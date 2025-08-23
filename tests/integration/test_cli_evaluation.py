"""
Integration tests for CLI evaluation workflows.
Tests the complete CLI evaluation functionality end-to-end.
"""

import pytest
import tempfile
import json
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from argparse import Namespace

from keisei.training.train import (
    run_evaluation_command,
    create_agent_info_from_checkpoint,
    add_evaluation_arguments
)
from keisei.config_schema import EvaluationConfig


class TestCLIEvaluationWorkflows:
    """Test CLI evaluation command workflows."""
    
    @pytest.mark.asyncio
    async def test_run_evaluation_command_basic(self):
        """Test basic evaluation command execution."""
        # Create temporary checkpoint file
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp_file:
            checkpoint_path = tmp_file.name
            
            # Create mock checkpoint data
            import torch
            mock_data = {"model_state_dict": {}, "metadata": {"step": 1000}}
            torch.save(mock_data, tmp_file.name)
        
        try:
            # Setup args
            args = Namespace(
                agent_checkpoint=checkpoint_path,
                config=None,
                strategy="single_opponent",
                num_games=5,
                opponent_type="random",
                opponent_checkpoint=None,
                device="cpu",
                wandb_log_eval=False,
                save_games=False,
                output_dir=None,
                run_name="test_cli_eval"
            )
            
            # Mock the evaluation manager and result - patch the import source
            with patch('keisei.evaluation.core_manager.EvaluationManager') as mock_manager_class:
                mock_manager = Mock()
                mock_result = Mock()
                mock_result.summary_stats = Mock()
                mock_result.summary_stats.win_rate = 0.65
                mock_result.summary_stats.total_games = 5
                mock_manager.evaluate_checkpoint_async = AsyncMock(return_value=mock_result)
                mock_manager_class.return_value = mock_manager
                
                # Execute
                result = await run_evaluation_command(args)
                
                # Verify
                assert result == mock_result
                mock_manager_class.assert_called_once()
                mock_manager.setup.assert_called_once()
                mock_manager.evaluate_checkpoint_async.assert_called_once_with(
                    checkpoint_path, None
                )
                
        finally:
            Path(checkpoint_path).unlink(missing_ok=True)
    
    @pytest.mark.asyncio
    async def test_run_evaluation_command_with_config_file(self):
        """Test evaluation command with config file."""
        # Create temporary files
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as checkpoint_file:
            checkpoint_path = checkpoint_file.name
            import torch
            torch.save({"model_state_dict": {}}, checkpoint_file.name)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as config_file:
            config_path = config_file.name
            config_file.write("""
evaluation:
  strategy: tournament
  num_games: 10
  opponent_type: heuristic
  save_games: true
""")
        
        try:
            args = Namespace(
                agent_checkpoint=checkpoint_path,
                config=config_path,
                strategy=None,  # Should use config value
                num_games=None,  # Should use config value
                opponent_type=None,  # Should use config value
                opponent_checkpoint=None,
                device="cpu",
                wandb_log_eval=False,
                save_games=None,  # Should use config value
                output_dir=None,
                run_name="test_config_eval"
            )
            
            # Mock evaluation components
            with patch('keisei.training.train.load_config') as mock_load_config:
                mock_config = Mock()
                mock_config.evaluation = EvaluationConfig(
                    strategy="tournament",
                    num_games=10,
                    opponent_type="heuristic",
                    save_games=True
                )
                mock_load_config.return_value = mock_config
                
                with patch('keisei.evaluation.core_manager.EvaluationManager') as mock_manager_class:
                    mock_manager = Mock()
                    mock_result = Mock()
                    mock_result.summary_stats = Mock()
                    mock_result.summary_stats.win_rate = 0.8
                    mock_manager.evaluate_checkpoint_async = AsyncMock(return_value=mock_result)
                    mock_manager_class.return_value = mock_manager
                    
                    # Execute
                    result = await run_evaluation_command(args)
                    
                    # Verify config was loaded and used
                    mock_load_config.assert_called_once_with(config_path, {})
                    mock_manager_class.assert_called_once()
                    
                    # Verify config values were applied
                    call_args = mock_manager_class.call_args
                    eval_config = call_args[1]['config']
                    assert eval_config.strategy == "tournament"
                    assert eval_config.num_games == 10
                    assert eval_config.opponent_type == "heuristic"
                    assert eval_config.save_games == True
                    
        finally:
            Path(checkpoint_path).unlink(missing_ok=True)
            Path(config_path).unlink(missing_ok=True)
    
    @pytest.mark.asyncio
    async def test_run_evaluation_command_with_save_results(self):
        """Test evaluation command with result saving."""
        # Create temporary checkpoint
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp_file:
            checkpoint_path = tmp_file.name
            import torch
            torch.save({"model_state_dict": {}}, tmp_file.name)
        
        # Create temporary output directory
        with tempfile.TemporaryDirectory() as output_dir:
            try:
                args = Namespace(
                    agent_checkpoint=checkpoint_path,
                    config=None,
                    strategy="single_opponent",
                    num_games=3,
                    opponent_type="random",
                    opponent_checkpoint=None,
                    device="cpu",
                    wandb_log_eval=False,
                    save_games=True,
                    output_dir=output_dir,
                    run_name="test_save_eval"
                )
                
                # Mock evaluation manager
                with patch('keisei.evaluation.core_manager.EvaluationManager') as mock_manager_class:
                    mock_manager = Mock()
                    mock_result = Mock()
                    mock_result.summary_stats = Mock()
                    mock_result.summary_stats.win_rate = 0.7
                    mock_result.summary_stats.total_games = 3
                    mock_result.model_dump = Mock(return_value={"win_rate": 0.7})
                    mock_manager.evaluate_checkpoint_async = AsyncMock(return_value=mock_result)
                    mock_manager_class.return_value = mock_manager
                    
                    # Execute
                    result = await run_evaluation_command(args)
                    
                    # Verify result file was created
                    output_path = Path(output_dir)
                    result_files = list(output_path.glob("test_save_eval_results.json"))
                    assert len(result_files) == 1
                    
                    # Verify result file content
                    with open(result_files[0], 'r') as f:
                        saved_data = json.load(f)
                    
                    assert saved_data["agent_checkpoint"] == checkpoint_path
                    assert "evaluation_config" in saved_data
                    assert "results" in saved_data
                    assert "timestamp" in saved_data
                    
            finally:
                Path(checkpoint_path).unlink(missing_ok=True)
    
    def test_create_agent_info_from_checkpoint_valid(self):
        """Test creating agent info from valid checkpoint."""
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp_file:
            checkpoint_path = tmp_file.name
            import torch
            torch.save({"model_state_dict": {}}, tmp_file.name)
        
        try:
            agent_info = create_agent_info_from_checkpoint(checkpoint_path)
            
            assert agent_info.name == Path(checkpoint_path).stem
            assert agent_info.checkpoint_path == checkpoint_path
            assert agent_info.metadata["source"] == "cli_evaluation"
            
        finally:
            Path(checkpoint_path).unlink(missing_ok=True)
    
    def test_create_agent_info_from_checkpoint_missing(self):
        """Test creating agent info from missing checkpoint."""
        nonexistent_path = "/this/path/does/not/exist.pt"
        
        with pytest.raises(FileNotFoundError):
            create_agent_info_from_checkpoint(nonexistent_path)
    
    def test_evaluation_argument_parsing(self):
        """Test that evaluation arguments are properly defined."""
        import argparse
        
        parser = argparse.ArgumentParser()
        add_evaluation_arguments(parser)
        
        # Test parsing valid arguments
        args = parser.parse_args([
            "--agent_checkpoint", "/path/to/model.pt",
            "--strategy", "tournament",
            "--num_games", "10",
            "--opponent_type", "heuristic",
            "--wandb_log_eval",
            "--save_games"
        ])
        
        assert args.agent_checkpoint == "/path/to/model.pt"
        assert args.strategy == "tournament"
        assert args.num_games == 10
        assert args.opponent_type == "heuristic"
        assert args.wandb_log_eval == True
        assert args.save_games == True
    
    @pytest.mark.asyncio
    async def test_evaluation_command_error_handling(self):
        """Test error handling in evaluation command."""
        # Setup args with invalid checkpoint
        args = Namespace(
            agent_checkpoint="/nonexistent/checkpoint.pt",
            config=None,
            strategy="single_opponent",
            num_games=1,
            opponent_type="random",
            opponent_checkpoint=None,
            device="cpu",
            wandb_log_eval=False,
            save_games=False,
            output_dir=None,
            run_name="test_error"
        )
        
        # Should raise error for missing checkpoint
        with pytest.raises(FileNotFoundError):
            await run_evaluation_command(args)
    
    @pytest.mark.asyncio
    async def test_evaluation_command_with_opponent_checkpoint(self):
        """Test evaluation command with opponent checkpoint."""
        # Create temporary checkpoint files
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as agent_file:
            agent_checkpoint = agent_file.name
            import torch
            torch.save({"model_state_dict": {}}, agent_file.name)
        
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as opponent_file:
            opponent_checkpoint = opponent_file.name
            torch.save({"model_state_dict": {}}, opponent_file.name)
        
        try:
            args = Namespace(
                agent_checkpoint=agent_checkpoint,
                config=None,
                strategy="single_opponent",
                num_games=2,
                opponent_type="ppo",
                opponent_checkpoint=opponent_checkpoint,
                device="cpu",
                wandb_log_eval=False,
                save_games=False,
                output_dir=None,
                run_name="test_opponent"
            )
            
            # Mock evaluation
            with patch('keisei.evaluation.core_manager.EvaluationManager') as mock_manager_class:
                mock_manager = Mock()
                mock_result = Mock()
                mock_result.summary_stats = Mock()
                mock_result.summary_stats.win_rate = 0.5
                mock_manager.evaluate_checkpoint_async = AsyncMock(return_value=mock_result)
                mock_manager_class.return_value = mock_manager
                
                # Execute
                result = await run_evaluation_command(args)
                
                # Verify opponent checkpoint was passed
                mock_manager.evaluate_checkpoint_async.assert_called_once_with(
                    agent_checkpoint, opponent_checkpoint
                )
                
        finally:
            Path(agent_checkpoint).unlink(missing_ok=True)
            Path(opponent_checkpoint).unlink(missing_ok=True)


class TestCLIIntegrationWithAsyncFixes:
    """Test CLI integration with async fixes."""
    
    @pytest.mark.asyncio
    async def test_cli_uses_async_evaluation_manager_methods(self):
        """Test that CLI uses the new async evaluation methods."""
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp_file:
            checkpoint_path = tmp_file.name
            import torch
            torch.save({"model_state_dict": {}}, tmp_file.name)
        
        try:
            args = Namespace(
                agent_checkpoint=checkpoint_path,
                config=None,
                strategy="single_opponent",
                num_games=1,
                opponent_type="random",
                opponent_checkpoint=None,
                device="cpu",
                wandb_log_eval=False,
                save_games=False,
                output_dir=None,
                run_name="test_async"
            )
            
            # Verify that evaluate_checkpoint_async is called (not the sync version)
            with patch('keisei.evaluation.core_manager.EvaluationManager') as mock_manager_class:
                mock_manager = Mock()
                mock_result = Mock()
                # Create a proper mock with numeric win_rate for formatting
                mock_summary_stats = Mock()
                mock_summary_stats.win_rate = 0.75  # Numeric value for formatting
                mock_result.summary_stats = mock_summary_stats
                mock_manager.evaluate_checkpoint_async = AsyncMock(return_value=mock_result)
                mock_manager_class.return_value = mock_manager
                
                # Execute
                await run_evaluation_command(args)
                
                # Verify async method was called
                mock_manager.evaluate_checkpoint_async.assert_called_once()
                
                # Verify sync method was NOT called (if it exists)
                if hasattr(mock_manager, 'evaluate_checkpoint'):
                    mock_manager.evaluate_checkpoint.assert_not_called()
                    
        finally:
            Path(checkpoint_path).unlink(missing_ok=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
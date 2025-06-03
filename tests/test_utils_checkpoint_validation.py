"""
test_utils_checkpoint_validation.py: Unit tests for checkpoint validation in utils.py
"""

import os
import tempfile
import pytest
import torch
from unittest.mock import patch, mock_open

from keisei.training.utils import find_latest_checkpoint, _validate_checkpoint


class TestCheckpointValidation:
    """Test checkpoint validation and corruption handling."""

    def test_validate_checkpoint_valid_file(self, tmp_path):
        """Test that _validate_checkpoint returns True for valid checkpoints."""
        # Create a valid checkpoint file
        checkpoint_path = tmp_path / "valid_checkpoint.pth"
        test_data = {"model_state_dict": {"layer.weight": torch.randn(10, 5)}}
        torch.save(test_data, checkpoint_path)

        # Test validation
        assert _validate_checkpoint(str(checkpoint_path)) is True

    def test_validate_checkpoint_corrupted_file(self, tmp_path):
        """Test that _validate_checkpoint returns False for corrupted checkpoints."""
        # Create a corrupted checkpoint file (invalid data)
        checkpoint_path = tmp_path / "corrupted_checkpoint.pth"
        with open(checkpoint_path, "wb") as f:
            f.write(b"invalid checkpoint data")

        # Test validation
        assert _validate_checkpoint(str(checkpoint_path)) is False

    def test_validate_checkpoint_missing_file(self):
        """Test that _validate_checkpoint returns False for missing files."""
        missing_path = "/nonexistent/path/checkpoint.pth"
        assert _validate_checkpoint(missing_path) is False

    def test_find_latest_checkpoint_with_valid_checkpoints(self, tmp_path):
        """Test that find_latest_checkpoint returns the latest valid checkpoint."""
        # Create multiple checkpoint files with different timestamps
        older_checkpoint = tmp_path / "checkpoint_1000.pth"
        newer_checkpoint = tmp_path / "checkpoint_2000.pth"
        
        test_data = {"model_state_dict": {"layer.weight": torch.randn(10, 5)}}
        torch.save(test_data, older_checkpoint)
        torch.save(test_data, newer_checkpoint)
        
        # Ensure newer has later modification time
        os.utime(newer_checkpoint, (1000, 1000))
        os.utime(older_checkpoint, (500, 500))

        result = find_latest_checkpoint(str(tmp_path))
        assert result == str(newer_checkpoint)

    def test_find_latest_checkpoint_with_corrupted_latest(self, tmp_path):
        """Test that find_latest_checkpoint skips corrupted checkpoints and returns valid one."""
        # Create valid older checkpoint and corrupted newer checkpoint
        older_checkpoint = tmp_path / "checkpoint_1000.pth"
        newer_corrupted = tmp_path / "checkpoint_2000.pth"
        
        # Create valid older checkpoint
        test_data = {"model_state_dict": {"layer.weight": torch.randn(10, 5)}}
        torch.save(test_data, older_checkpoint)
        
        # Create corrupted newer checkpoint
        with open(newer_corrupted, "wb") as f:
            f.write(b"corrupted data")
        
        # Ensure newer has later modification time
        os.utime(newer_corrupted, (1000, 1000))
        os.utime(older_checkpoint, (500, 500))

        result = find_latest_checkpoint(str(tmp_path))
        assert result == str(older_checkpoint)

    def test_find_latest_checkpoint_all_corrupted(self, tmp_path):
        """Test that find_latest_checkpoint returns None when all checkpoints are corrupted."""
        # Create multiple corrupted checkpoint files
        corrupted1 = tmp_path / "checkpoint_1000.pth"
        corrupted2 = tmp_path / "checkpoint_2000.pth"
        
        with open(corrupted1, "wb") as f:
            f.write(b"corrupted data 1")
        with open(corrupted2, "wb") as f:
            f.write(b"corrupted data 2")

        result = find_latest_checkpoint(str(tmp_path))
        assert result is None

    def test_find_latest_checkpoint_no_checkpoints(self, tmp_path):
        """Test that find_latest_checkpoint returns None when no checkpoints exist."""
        result = find_latest_checkpoint(str(tmp_path))
        assert result is None

    def test_find_latest_checkpoint_directory_not_exist(self):
        """Test that find_latest_checkpoint handles non-existent directories gracefully."""
        result = find_latest_checkpoint("/nonexistent/directory")
        assert result is None

    def test_validate_checkpoint_handles_various_exceptions(self, tmp_path):
        """Test that _validate_checkpoint handles various types of checkpoint corruption."""
        # Test EOFError (truncated file)
        truncated_checkpoint = tmp_path / "truncated.pth"
        with open(truncated_checkpoint, "wb") as f:
            f.write(b"PK")  # Just partial file header
        
        assert _validate_checkpoint(str(truncated_checkpoint)) is False

    @patch("builtins.print")
    def test_validation_error_messages(self, mock_print, tmp_path):
        """Test that validation functions print appropriate error messages."""
        # Test corrupted checkpoint error message
        corrupted_path = tmp_path / "corrupted.pth"
        with open(corrupted_path, "wb") as f:
            f.write(b"invalid data")
        
        result = _validate_checkpoint(str(corrupted_path))
        assert result is False
        
        # Verify error message was printed
        mock_print.assert_called()
        call_args = mock_print.call_args[0][0]
        assert "Corrupted checkpoint" in call_args
        assert str(corrupted_path) in call_args

    @patch("builtins.print")
    def test_find_latest_checkpoint_all_corrupted_message(self, mock_print, tmp_path):
        """Test that find_latest_checkpoint prints message when all checkpoints are corrupted."""
        # Create corrupted checkpoint
        corrupted = tmp_path / "corrupted.pth"
        with open(corrupted, "wb") as f:
            f.write(b"corrupted data")

        result = find_latest_checkpoint(str(tmp_path))
        assert result is None
        
        # Verify the all-corrupted message was printed
        mock_print.assert_any_call(
            "All checkpoint files in directory are corrupted or unreadable",
            file=pytest.importorskip("sys").stderr
        )

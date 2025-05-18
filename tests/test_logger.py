"""
Unit test for TrainingLogger in utils.py
"""

from keisei.utils import TrainingLogger


def test_training_logger(tmp_path):
    log_path = tmp_path / "test.log"
    logger = TrainingLogger(str(log_path), also_stdout=False)
    logger.log("Test message 1")
    logger.log("Test message 2")
    logger.close()
    with open(log_path, encoding="utf-8") as f:
        lines = f.readlines()
    assert any("Test message 1" in line for line in lines)
    assert any("Test message 2" in line for line in lines)

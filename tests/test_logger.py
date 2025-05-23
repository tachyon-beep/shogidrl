"""
Unit test for TrainingLogger in utils.py
"""

from keisei.utils import EvaluationLogger, TrainingLogger


def test_training_logger(tmp_path):
    log_path = tmp_path / "test.log"
    with TrainingLogger(str(log_path), also_stdout=False) as logger:
        logger.log("Test message 1")
        logger.log("Test message 2")
    with open(log_path, encoding="utf-8") as f:
        lines = f.readlines()
    assert any("Test message 1" in line for line in lines)
    assert any("Test message 2" in line for line in lines)


def test_evaluation_logger(tmp_path):
    log_path = tmp_path / "evaluation_test.log"
    with EvaluationLogger(str(log_path), also_stdout=False) as logger:
        logger.log_evaluation_result(
            iteration=1,
            opponent_name="TestOpponent",
            win_rate=0.75,
            avg_game_length=50.5,
            num_games=100,
        )
        logger.log_custom_message("Custom evaluation message")

    with open(log_path, encoding="utf-8") as f:
        lines = f.readlines()

    assert any("Iteration: 1, Opponent: TestOpponent" in line for line in lines)
    assert any("Win Rate: 0.75" in line for line in lines)
    assert any("Avg Game Length: 50.50" in line for line in lines)  # Check formatting
    assert any("Games: 100" in line for line in lines)
    assert any("Custom evaluation message" in line for line in lines)

import pytest
from keisei.training.display_components import ShogiBoard
from wcwidth import wcswidth

class DummyBoard:
    def __init__(self):
        self.board = [[None for _ in range(9)] for _ in range(9)]

@pytest.mark.parametrize("use_unicode", [True, False])
def test_shogi_board_line_width_alignment(use_unicode):
    board = DummyBoard()
    shogi_board = ShogiBoard(use_unicode=use_unicode)
    ascii_board = shogi_board._generate_ascii_board(board)
    line_widths = [wcswidth(line) for line in ascii_board.splitlines()]
    assert len(set(line_widths)) == 1

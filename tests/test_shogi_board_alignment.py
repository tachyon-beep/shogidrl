import pytest
from keisei.training.display_components import ShogiBoard
from wcwidth import wcswidth

@pytest.mark.parametrize("use_unicode", [True, False])
def test_shogi_board_padding_width(use_unicode):
    shogi_board = ShogiBoard(use_unicode=use_unicode)
    if use_unicode:
        symbols = ["歩", "香", "桂", "銀", "金", "角", "飛", "王", "と", "成香", "成桂", "成銀", "馬", "竜", "・"]
    else:
        symbols = ["P", "L", "N", "S", "G", "B", "R", "K", "+P", "+L", "+N", "+S", "+B", "+R", "."]

    for sym in symbols:
        padded = shogi_board._pad_symbol(sym)
        assert wcswidth(padded) == shogi_board.cell_width

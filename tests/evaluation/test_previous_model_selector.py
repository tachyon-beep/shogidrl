from pathlib import Path

from keisei.training.previous_model_selector import PreviousModelSelector


def test_previous_model_selector_basic(tmp_path):
    selector = PreviousModelSelector(pool_size=2)
    ck1 = tmp_path / "ck1.pth"
    ck2 = tmp_path / "ck2.pth"
    ck1.write_text("a")
    ck2.write_text("b")
    selector.add_checkpoint(ck1)
    selector.add_checkpoint(str(ck2))
    # Both checkpoints should be stored
    all_ckpts = selector.get_all()
    assert Path(ck1) in all_ckpts and Path(ck2) in all_ckpts
    # Random selection returns one of them
    chosen = selector.get_random_checkpoint()
    assert chosen in {ck1, ck2}
    # Adding third should evict first
    ck3 = tmp_path / "ck3.pth"
    ck3.write_text("c")
    selector.add_checkpoint(ck3)
    assert len(selector.get_all()) == 2
    assert Path(ck3) in selector.get_all()

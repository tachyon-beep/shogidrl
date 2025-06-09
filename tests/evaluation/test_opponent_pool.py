from pathlib import Path

from keisei.evaluation.opponents import OpponentPool


def test_pool_add_sample_and_evict(tmp_path):
    pool = OpponentPool(pool_size=2, elo_registry_path=str(tmp_path / "elo.json"))
    ck1 = tmp_path / "ck1.pth"
    ck2 = tmp_path / "ck2.pth"
    ck1.write_text("a")
    ck2.write_text("b")

    pool.add_checkpoint(ck1)
    pool.add_checkpoint(str(ck2))
    assert set(pool.get_all()) == {ck1, ck2}
    assert pool.sample() in {ck1, ck2}

    ck3 = tmp_path / "ck3.pth"
    ck3.write_text("c")
    pool.add_checkpoint(ck3)
    assert len(pool.get_all()) == 2
    assert ck1 not in pool.get_all()


def test_pool_champion_rating(tmp_path):
    pool = OpponentPool(pool_size=3, elo_registry_path=str(tmp_path / "elo.json"))
    ck1 = tmp_path / "a.pth"
    ck2 = tmp_path / "b.pth"
    ck1.write_text("a")
    ck2.write_text("b")
    pool.add_checkpoint(ck1)
    pool.add_checkpoint(ck2)
    # Update Elo so ck2 becomes champion
    pool.update_ratings(ck2.name, ck1.name, ["agent_win"])
    assert pool.champion() == ck2


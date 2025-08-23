from datetime import datetime

import pytest

from keisei.evaluation.core import (
    AgentInfo,
    EvaluationContext,
    GameResult,
    OpponentInfo,
    SummaryStats,
    EvaluationConfig,
    create_evaluation_config,
)


def test_context_creation_and_serialization():
    config = create_evaluation_config(
        strategy="single_opponent",
        num_games=1
    )
    ctx = EvaluationContext(
        session_id="sess1",
        timestamp=datetime(2025, 1, 1),
        agent_info=AgentInfo(name="AgentA"),
        configuration=config,
        environment_info={"device": "cpu"},
        metadata={"flag": True},
    )

    data = ctx.to_dict()
    new_ctx = EvaluationContext.from_dict(data, config)

    assert new_ctx.session_id == ctx.session_id
    assert new_ctx.agent_info.name == "AgentA"
    assert new_ctx.metadata["flag"] is True
    assert new_ctx.configuration.strategy == config.strategy


def test_summary_stats_from_games():
    agent = AgentInfo(name="AgentA")
    opp = OpponentInfo(name="Opp", type="random")
    games = [
        GameResult(
            game_id="g1",
            winner=0,
            moves_count=10,
            duration_seconds=1.0,
            agent_info=agent,
            opponent_info=opp,
        ),
        GameResult(
            game_id="g2",
            winner=1,
            moves_count=20,
            duration_seconds=2.0,
            agent_info=agent,
            opponent_info=opp,
        ),
        GameResult(
            game_id="g3",
            winner=None,
            moves_count=15,
            duration_seconds=1.5,
            agent_info=agent,
            opponent_info=opp,
        ),
    ]

    stats = SummaryStats.from_games(games)
    assert stats.total_games == 3
    assert stats.agent_wins == 1
    assert stats.opponent_wins == 1
    assert stats.draws == 1
    assert stats.total_moves == 45
    assert stats.avg_game_length == 15
    assert stats.avg_duration_seconds == pytest.approx(1.5)
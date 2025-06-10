# filepath: /home/john/keisei/tests/evaluation/strategies/test_tournament_evaluator.py
import asyncio
import math  # For pytest.approx, though often not needed if pytest is imported
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
import torch

from keisei.constants import GameTerminationReason  # Added import
from keisei.evaluation.core import (
    AgentInfo,
    BaseEvaluator,
    EvaluationContext,
    EvaluationResult,
    GameResult,
    OpponentInfo,
    SummaryStats,
    TournamentConfig,
)
from keisei.evaluation.strategies.tournament import (
    TERMINATION_REASON_ACTION_SELECTION_ERROR,
    TERMINATION_REASON_EVAL_STEP_ERROR,
    TERMINATION_REASON_GAME_ENDED_UNSPECIFIED,
    TERMINATION_REASON_ILLEGAL_MOVE,
    TERMINATION_REASON_MAX_MOVES,
    TERMINATION_REASON_MOVE_EXECUTION_ERROR,
    TERMINATION_REASON_NO_LEGAL_MOVES_UNDETERMINED,
    TERMINATION_REASON_UNKNOWN_LOOP_TERMINATION,
    TournamentEvaluator,
)
from keisei.shogi.shogi_core_definitions import Color
from keisei.shogi.shogi_game import ShogiGame
from keisei.utils import PolicyOutputMapper


# Note: Using @pytest.mark.asyncio for async tests instead of custom wrapper


@pytest.fixture
def mock_tournament_config():
    config = MagicMock(spec=TournamentConfig)
    config.name = "TestTournament"
    config.num_games = 10
    config.num_games_per_opponent = 2
    config.opponent_pool_config = []
    config.default_device = "cpu"
    config.input_channels = 46
    config.max_moves_per_game = 100
    config.log_level = "INFO"
    config.evaluation_strategy = "tournament"
    config.output_dir = "/tmp/eval_output"
    config.save_results = True
    config.save_replays = False
    config.custom_params = {}
    return config


@pytest.fixture
def mock_agent_info():
    agent_info = MagicMock(spec=AgentInfo)
    agent_info.name = "TestAgent"
    agent_info.checkpoint_path = "/path/to/agent.ptk"
    agent_info.type = "ppo_agent"
    agent_info.to_dict = MagicMock(
        return_value={
            "name": "TestAgent",
            "checkpoint_path": "/path/to/agent.ptk",
            "type": "ppo_agent",
        }
    )
    return agent_info


@pytest.fixture
def mock_opponent_info():
    opponent_info = MagicMock(spec=OpponentInfo)
    opponent_info.name = "TestOpponent1"
    opponent_info.type = "random"
    opponent_info.checkpoint_path = None
    opponent_info.metadata = {}
    opponent_info.to_dict = MagicMock(
        return_value={"name": "TestOpponent1", "type": "random", "metadata": {}}
    )
    return opponent_info


@pytest.fixture
def mock_evaluation_context(mock_tournament_config, mock_agent_info):
    context = MagicMock(spec=EvaluationContext)
    context.session_id = "test_session_123"
    context.configuration = mock_tournament_config
    context.agent_info = mock_agent_info
    context.timestamp = MagicMock()
    context.environment_info = {}
    context.metadata = {}
    return context


class TestTournamentEvaluator:

    def test_init(self, mock_tournament_config):
        evaluator = TournamentEvaluator(mock_tournament_config)
        assert evaluator.config == mock_tournament_config
        assert (
            evaluator.logger is not None
        ), "Logger should be initialized by BaseEvaluator"
        assert isinstance(evaluator.policy_mapper, PolicyOutputMapper)

    @pytest.mark.asyncio
    async def test_load_tournament_opponents_empty_config(
        self, mock_tournament_config, mock_evaluation_context
    ):
        mock_tournament_config.opponent_pool_config = []
        evaluator = TournamentEvaluator(mock_tournament_config)

        with patch(
            "keisei.evaluation.strategies.tournament.logger", MagicMock()
        ) as mock_logger:
            opponents = await evaluator._load_tournament_opponents()
            assert opponents == []
            mock_logger.warning.assert_called_once()

    @pytest.mark.asyncio
    async def test_load_tournament_opponents_valid_config(
        self, mock_tournament_config, mock_evaluation_context
    ):
        opponent_data_1 = {"name": "Opp1", "type": "random", "metadata": {"level": 1}}
        opponent_data_2 = {
            "name": "Opp2",
            "type": "ppo_agent",
            "checkpoint_path": "/path/opp2.ptk",
        }
        mock_tournament_config.opponent_pool_config = [opponent_data_1, opponent_data_2]

        evaluator = TournamentEvaluator(mock_tournament_config)

        with patch("keisei.evaluation.strategies.tournament.logger", MagicMock()):
            opponents = await evaluator._load_tournament_opponents()
            assert len(opponents) == 2
            assert opponents[0].name == "Opp1"
            assert opponents[0].type == "random"
            assert opponents[0].metadata == {"level": 1}
            assert opponents[1].name == "Opp2"
            assert opponents[1].type == "ppo_agent"
            assert opponents[1].checkpoint_path == "/path/opp2.ptk"
            assert opponents[1].metadata == {}

    @pytest.mark.asyncio
    async def test_load_tournament_opponents_already_opponent_info(
        self, mock_tournament_config, mock_evaluation_context
    ):
        opp_info_1 = OpponentInfo(
            name="PredefinedOpp1", type="heuristic", metadata={"id": "pre1"}
        )
        opponent_data_2 = {"name": "Opp2", "type": "random"}
        mock_tournament_config.opponent_pool_config = [opp_info_1, opponent_data_2]

        evaluator = TournamentEvaluator(mock_tournament_config)
        with patch("keisei.evaluation.strategies.tournament.logger", MagicMock()):
            opponents = await evaluator._load_tournament_opponents()
            assert len(opponents) == 2
            assert opponents[0] == opp_info_1
            assert opponents[1].name == "Opp2"
            assert opponents[1].type == "random"

    @pytest.mark.asyncio
    async def test_load_tournament_opponents_malformed_entry(
        self, mock_tournament_config, mock_evaluation_context
    ):
        opponent_data_1 = {"name": "Opp1", "type": "random"}
        malformed_data = "not_a_dict_or_opponent_info"
        mock_tournament_config.opponent_pool_config = [opponent_data_1, malformed_data]

        evaluator = TournamentEvaluator(mock_tournament_config)
        with patch(
            "keisei.evaluation.strategies.tournament.logger", MagicMock()
        ) as mock_logger:
            opponents = await evaluator._load_tournament_opponents()
            assert len(opponents) == 1
            assert opponents[0].name == "Opp1"
            mock_logger.warning.assert_called_once_with(
                "Unsupported opponent config format: %s at index %d. Skipping.",
                "str",
                1,
            )

    @pytest.mark.asyncio
    async def test_load_tournament_opponents_error_in_processing_entry(
        self, mock_tournament_config, mock_evaluation_context
    ):
        opponent_data_1 = {"name": "Opp1", "type": "random"}
        problematic_data = {
            "name": "ProblemOpp",
            "type": "special_type_that_breaks_init",
        }
        mock_tournament_config.opponent_pool_config = [
            opponent_data_1,
            problematic_data,
        ]

        evaluator = TournamentEvaluator(mock_tournament_config)

        with patch(
            "keisei.evaluation.strategies.tournament.logger", MagicMock()
        ) as mock_logger:
            original_opponent_info_init = OpponentInfo.__init__

            def mock_init(self_obj, *args, **kwargs_init):
                if kwargs_init.get("name") == "ProblemOpp":
                    raise ValueError("Simulated init error")
                original_opponent_info_init(self_obj, *args, **kwargs_init)

            with patch("keisei.evaluation.core.OpponentInfo.__init__", mock_init):
                opponents = await evaluator._load_tournament_opponents()

            assert len(opponents) == 1
            assert opponents[0].name == "Opp1"

            assert mock_logger.error.call_count == 1
            args, _ = mock_logger.error.call_args
            assert "Failed to load opponent from config data at index 1" in args[0]
            assert "Simulated init error" in str(args[2])

    def test_validate_config_valid(self, mock_tournament_config):
        mock_tournament_config.opponent_pool_config = [
            {"name": "Opp1", "type": "random"}
        ]
        evaluator = TournamentEvaluator(mock_tournament_config)
        with patch.object(BaseEvaluator, "validate_config", return_value=True):
            assert evaluator.validate_config() is True

    def test_validate_config_missing_opponent_pool(self, mock_tournament_config):
        mock_tournament_config.opponent_pool_config = None
        evaluator = TournamentEvaluator(mock_tournament_config)
        with (
            patch.object(BaseEvaluator, "validate_config", return_value=True),
            patch(
                "keisei.evaluation.strategies.tournament.logger", MagicMock()
            ) as mock_logger,
        ):
            assert evaluator.validate_config() is True
            mock_logger.warning.assert_called_once()
            assert (
                "TournamentConfig.opponent_pool_config is missing or not a list"
                in mock_logger.warning.call_args[0][0]
            )

    def test_validate_config_opponent_pool_not_list(self, mock_tournament_config):
        mock_tournament_config.opponent_pool_config = {"not": "a list"}
        evaluator = TournamentEvaluator(mock_tournament_config)
        with (
            patch.object(BaseEvaluator, "validate_config", return_value=True),
            patch(
                "keisei.evaluation.strategies.tournament.logger", MagicMock()
            ) as mock_logger,
        ):
            assert evaluator.validate_config() is True
            mock_logger.warning.assert_called_once()
            assert (
                "TournamentConfig.opponent_pool_config is missing or not a list"
                in mock_logger.warning.call_args[0][0]
            )

    def test_validate_config_base_invalid(self, mock_tournament_config):
        evaluator = TournamentEvaluator(mock_tournament_config)
        with patch.object(BaseEvaluator, "validate_config", return_value=False):
            assert evaluator.validate_config() is False

    @pytest.mark.asyncio
    async def test_evaluate_no_opponents(
        self, mock_tournament_config, mock_agent_info, mock_evaluation_context
    ):
        mock_tournament_config.opponent_pool_config = []
        evaluator = TournamentEvaluator(mock_tournament_config)

        evaluator.setup_context = MagicMock(return_value=mock_evaluation_context)
        evaluator.log_evaluation_start = MagicMock()
        evaluator.log_evaluation_complete = MagicMock()
        evaluator._load_tournament_opponents = AsyncMock(return_value=[])

        with patch(
            "keisei.evaluation.core.SummaryStats.from_games",
            MagicMock(return_value=MagicMock(spec=SummaryStats)),
        ) as mock_summary_stats_from_games:
            result = await evaluator.evaluate(mock_agent_info, mock_evaluation_context)

            assert isinstance(result, EvaluationResult)
            assert result.games == []
            assert result.errors == ["No opponents loaded for tournament."]
            assert result.analytics_data == {"tournament_specific_analytics": {}}
            mock_summary_stats_from_games.assert_called_once_with([])
            evaluator.log_evaluation_start.assert_called_once()
            evaluator.log_evaluation_complete.assert_called_once()

    @pytest.mark.asyncio
    async def test_calculate_tournament_standings_no_games(
        self, mock_tournament_config, mock_agent_info
    ):
        evaluator = TournamentEvaluator(mock_tournament_config)
        opp1_info = OpponentInfo(name="Opp1", type="random")
        opp2_info = OpponentInfo(name="Opp2", type="random")

        standings = evaluator._calculate_tournament_standings(
            [], [opp1_info, opp2_info], mock_agent_info
        )

        assert standings["overall_tournament_stats"]["total_games"] == 0
        assert standings["overall_tournament_stats"]["agent_total_wins"] == 0
        assert standings["per_opponent_results"]["Opp1"]["played"] == 0
        assert standings["per_opponent_results"]["Opp2"]["played"] == 0

    @pytest.mark.asyncio
    async def test_calculate_tournament_standings_with_games(
        self, mock_tournament_config, mock_agent_info
    ):
        evaluator = TournamentEvaluator(mock_tournament_config)

        agent_info_mock = mock_agent_info
        opp1_info = OpponentInfo(name="Opp1", type="random")
        opp1_info.to_dict = MagicMock(return_value={"name": "Opp1", "type": "random"})
        opp2_info = OpponentInfo(name="Opp2", type="random")  # Corrected instantiation
        opp2_info.to_dict = MagicMock(return_value={"name": "Opp2", "type": "random"})

        game1_vs_opp1 = GameResult(
            game_id="g1",
            agent_info=agent_info_mock,
            opponent_info=opp1_info,
            winner=0,
            moves_count=10,
            duration_seconds=1.0,
            metadata={},
        )
        game2_vs_opp1 = GameResult(
            game_id="g2",
            agent_info=agent_info_mock,
            opponent_info=opp1_info,
            winner=1,
            moves_count=10,
            duration_seconds=1.0,
            metadata={},
        )
        game3_vs_opp2 = GameResult(
            game_id="g3",
            agent_info=agent_info_mock,
            opponent_info=opp2_info,
            winner=None,
            moves_count=10,
            duration_seconds=1.0,
            metadata={},
        )

        results = [game1_vs_opp1, game2_vs_opp1, game3_vs_opp2]
        opponents = [opp1_info, opp2_info]

        standings = evaluator._calculate_tournament_standings(
            results, opponents, agent_info_mock
        )

        assert standings["overall_tournament_stats"]["total_games"] == 3
        assert standings["overall_tournament_stats"]["agent_total_wins"] == 1
        assert standings["overall_tournament_stats"]["agent_total_losses"] == 1
        assert standings["overall_tournament_stats"]["agent_total_draws"] == 1
        assert standings["overall_tournament_stats"][
            "agent_overall_win_rate"
        ] == pytest.approx(1 / 3)

        assert standings["per_opponent_results"]["Opp1"]["played"] == 2
        assert standings["per_opponent_results"]["Opp1"]["wins"] == 1
        assert standings["per_opponent_results"]["Opp1"]["losses"] == 1
        assert standings["per_opponent_results"]["Opp1"]["draws"] == 0
        assert standings["per_opponent_results"]["Opp1"]["win_rate"] == pytest.approx(
            0.5
        )

        assert standings["per_opponent_results"]["Opp2"]["played"] == 1
        assert standings["per_opponent_results"]["Opp2"]["wins"] == 0
        assert standings["per_opponent_results"]["Opp2"]["losses"] == 0
        assert standings["per_opponent_results"]["Opp2"]["draws"] == 1
        assert standings["per_opponent_results"]["Opp2"]["win_rate"] == pytest.approx(
            0.0
        )

    @pytest.mark.asyncio
    async def test_play_games_against_opponent(
        self,
        mock_tournament_config,
        mock_agent_info,
        mock_opponent_info,
        mock_evaluation_context,
    ):
        evaluator = TournamentEvaluator(mock_tournament_config)
        num_games_to_play = 2

        mock_game_result_sente = MagicMock(spec=GameResult)
        mock_game_result_gote = MagicMock(spec=GameResult)
        evaluator.evaluate_step = AsyncMock(
            side_effect=[mock_game_result_sente, mock_game_result_gote]
        )

        def opponent_info_from_dict_side_effect(data_dict):
            new_mock_opp = MagicMock(spec=OpponentInfo)
            new_mock_opp.name = data_dict.get("name")
            new_mock_opp.type = data_dict.get("type")
            new_mock_opp.metadata = data_dict.get("metadata", {}).copy()
            new_mock_opp.to_dict = MagicMock(return_value=data_dict)
            return new_mock_opp

        with patch(
            "keisei.evaluation.core.OpponentInfo.from_dict",
            side_effect=opponent_info_from_dict_side_effect,
        ) as mock_from_dict:
            results, errors = await evaluator._play_games_against_opponent(
                mock_agent_info,
                mock_opponent_info,
                num_games_to_play,
                mock_evaluation_context,
            )

        assert len(results) == 2
        assert results[0] == mock_game_result_sente
        assert results[1] == mock_game_result_gote
        assert errors == []

        assert evaluator.evaluate_step.call_count == 2

        call_args_list = evaluator.evaluate_step.call_args_list

        args_sente, _ = call_args_list[0]
        assert args_sente[0] == mock_agent_info
        assert args_sente[1].metadata["agent_plays_sente_in_eval_step"] is True
        assert args_sente[2] == mock_evaluation_context

        args_gote, _ = call_args_list[1]
        assert args_gote[0] == mock_agent_info
        assert args_gote[1].metadata["agent_plays_sente_in_eval_step"] is False
        assert args_gote[2] == mock_evaluation_context

        assert mock_from_dict.call_count == num_games_to_play

    @pytest.mark.asyncio
    async def test_play_games_against_opponent_eval_step_error(
        self,
        mock_tournament_config,
        mock_agent_info,
        mock_opponent_info,
        mock_evaluation_context,
    ):
        evaluator = TournamentEvaluator(mock_tournament_config)
        num_games_to_play = 1

        evaluator.evaluate_step = AsyncMock(
            side_effect=Exception("Simulated game error")
        )

        with patch(
            "keisei.evaluation.core.OpponentInfo.from_dict",
            return_value=mock_opponent_info,
        ):
            results, errors = await evaluator._play_games_against_opponent(
                mock_agent_info,
                mock_opponent_info,
                num_games_to_play,
                mock_evaluation_context,
            )

        assert len(results) == 0
        assert len(errors) == 1
        assert "Error during game orchestration" in errors[0]
        assert "Simulated game error" in errors[0]

    @pytest.mark.asyncio
    async def test_evaluate_step_successful_game_agent_sente(
        self,
        mock_tournament_config,
        mock_agent_info,
        mock_opponent_info,
        mock_evaluation_context,
    ):
        evaluator = TournamentEvaluator(mock_tournament_config)

        mock_opponent_info.metadata = {"agent_plays_sente_in_eval_step": True}

        mock_agent_entity = AsyncMock()
        mock_opponent_entity = AsyncMock()
        evaluator._game_load_evaluation_entity = AsyncMock(
            side_effect=[mock_agent_entity, mock_opponent_entity]
        )

        game_outcome = {
            "winner": 0,
            "moves_count": 30,
            "termination_reason": "Checkmate",
        }
        evaluator._game_run_game_loop = AsyncMock(return_value=game_outcome)

        with patch("time.time", side_effect=[100.0, 105.0]):
            result = await evaluator.evaluate_step(
                mock_agent_info, mock_opponent_info, mock_evaluation_context
            )

        assert isinstance(result, GameResult)
        assert result.winner == 0
        assert result.moves_count == 30
        assert result.duration_seconds == pytest.approx(5.0)
        assert result.metadata["agent_color"] == "Sente"
        assert result.metadata["termination_reason"] == "Checkmate"
        assert "agent_plays_sente_in_eval_step" not in result.metadata

        evaluator._game_load_evaluation_entity.assert_any_call(
            mock_agent_info, "cpu", 46
        )
        evaluator._game_load_evaluation_entity.assert_any_call(
            mock_opponent_info, "cpu", 46
        )
        evaluator._game_run_game_loop.assert_called_once_with(
            mock_agent_entity, mock_opponent_entity, mock_evaluation_context
        )

    @pytest.mark.asyncio
    async def test_evaluate_step_successful_game_agent_gote(
        self,
        mock_tournament_config,
        mock_agent_info,
        mock_opponent_info,
        mock_evaluation_context,
    ):
        evaluator = TournamentEvaluator(mock_tournament_config)

        mock_opponent_info.metadata = {"agent_plays_sente_in_eval_step": False}

        mock_agent_entity = AsyncMock()
        mock_opponent_entity = AsyncMock()
        evaluator._game_load_evaluation_entity = AsyncMock(
            side_effect=[mock_agent_entity, mock_opponent_entity]
        )

        game_outcome = {
            "winner": 0,
            "moves_count": 40,
            "termination_reason": "Resignation",
        }
        evaluator._game_run_game_loop = AsyncMock(return_value=game_outcome)

        with patch("time.time", side_effect=[200.0, 210.0]):
            result = await evaluator.evaluate_step(
                mock_agent_info, mock_opponent_info, mock_evaluation_context
            )

        assert result.winner == 1
        assert result.moves_count == 40
        assert result.duration_seconds == pytest.approx(10.0)
        assert result.metadata["agent_color"] == "Gote"
        assert result.metadata["termination_reason"] == "Resignation"
        evaluator._game_run_game_loop.assert_called_once_with(
            mock_opponent_entity, mock_agent_entity, mock_evaluation_context
        )

    @pytest.mark.asyncio
    async def test_evaluate_step_game_loop_error(
        self,
        mock_tournament_config,
        mock_agent_info,
        mock_opponent_info,
        mock_evaluation_context,
    ):
        evaluator = TournamentEvaluator(mock_tournament_config)
        mock_opponent_info.metadata = {"agent_plays_sente_in_eval_step": True}

        evaluator._game_load_evaluation_entity = AsyncMock(
            side_effect=[AsyncMock(), AsyncMock()]
        )
        evaluator._game_run_game_loop = AsyncMock(side_effect=Exception("Loop error!"))

        with patch("time.time", side_effect=[300.0, 301.0]):
            result = await evaluator.evaluate_step(
                mock_agent_info, mock_opponent_info, mock_evaluation_context
            )

        assert result.winner is None
        assert result.moves_count == 0
        assert result.duration_seconds == pytest.approx(1.0)
        assert result.metadata["error"] == "Loop error!"
        assert (
            TERMINATION_REASON_EVAL_STEP_ERROR in result.metadata["termination_reason"]
        )
        assert "Loop error!" in result.metadata["termination_reason"]

    @pytest.mark.asyncio
    async def test_evaluate_step_load_entity_error(
        self,
        mock_tournament_config,
        mock_agent_info,
        mock_opponent_info,
        mock_evaluation_context,
    ):
        evaluator = TournamentEvaluator(mock_tournament_config)
        mock_opponent_info.metadata = {"agent_plays_sente_in_eval_step": True}

        evaluator._game_load_evaluation_entity = AsyncMock(
            side_effect=ValueError("Load failed!")
        )
        evaluator._game_run_game_loop = AsyncMock()

        with patch("time.time", side_effect=[400.0, 401.0]):
            result = await evaluator.evaluate_step(
                mock_agent_info, mock_opponent_info, mock_evaluation_context
            )

        assert result.winner is None
        assert result.moves_count == 0
        assert result.metadata["error"] == "Load failed!"
        assert (
            TERMINATION_REASON_EVAL_STEP_ERROR in result.metadata["termination_reason"]
        )
        assert "Load failed!" in result.metadata["termination_reason"]
        evaluator._game_run_game_loop.assert_not_called()

    @pytest.mark.asyncio
    async def test_evaluate_full_run_calculates_num_games_per_opponent_dynamically(
        self, mock_tournament_config, mock_agent_info, mock_evaluation_context
    ):
        mock_tournament_config.num_games_per_opponent = None
        mock_tournament_config.num_games = 6
        opp_info1 = OpponentInfo(name="DynOpp1", type="random")
        opp_info2 = OpponentInfo(name="DynOpp2", type="random")

        evaluator = TournamentEvaluator(mock_tournament_config)
        evaluator.setup_context = MagicMock(return_value=mock_evaluation_context)
        evaluator.log_evaluation_start = MagicMock()
        evaluator.log_evaluation_complete = MagicMock()
        evaluator._load_tournament_opponents = AsyncMock(
            return_value=[opp_info1, opp_info2]
        )

        evaluator._play_games_against_opponent = AsyncMock(return_value=([], []))
        evaluator._calculate_tournament_standings = MagicMock(return_value={})

        with patch(
            "keisei.evaluation.core.SummaryStats.from_games",
            MagicMock(return_value=MagicMock(spec=SummaryStats)),
        ):
            await evaluator.evaluate(mock_agent_info, mock_evaluation_context)

        expected_games_per_opponent = 3
        evaluator._play_games_against_opponent.assert_any_call(
            mock_agent_info,
            opp_info1,
            expected_games_per_opponent,
            mock_evaluation_context,
        )
        evaluator._play_games_against_opponent.assert_any_call(
            mock_agent_info,
            opp_info2,
            expected_games_per_opponent,
            mock_evaluation_context,
        )
        assert evaluator._play_games_against_opponent.call_count == 2

    @pytest.mark.asyncio
    async def test_evaluate_full_run_with_opponents_and_games(
        self,
        mock_tournament_config,
        mock_agent_info,
        mock_evaluation_context,
        mock_opponent_info,
    ):
        mock_tournament_config.num_games_per_opponent = 1

        opp1 = OpponentInfo(name="Opp1Evaluate", type="random", metadata={})
        opp1.to_dict = MagicMock(
            return_value={"name": "Opp1Evaluate", "type": "random", "metadata": {}}
        )
        opp2 = OpponentInfo(name="Opp2Evaluate", type="random", metadata={})
        opp2.to_dict = MagicMock(
            return_value={"name": "Opp2Evaluate", "type": "random", "metadata": {}}
        )

        mock_opponents_list = [opp1, opp2]

        evaluator = TournamentEvaluator(mock_tournament_config)
        evaluator.setup_context = MagicMock(return_value=mock_evaluation_context)
        evaluator.log_evaluation_start = MagicMock()
        evaluator.log_evaluation_complete = MagicMock()
        evaluator._load_tournament_opponents = AsyncMock(
            return_value=mock_opponents_list
        )

        mock_game_res1 = GameResult(
            game_id="g1e",
            agent_info=mock_agent_info,
            opponent_info=opp1,
            winner=0,
            moves_count=1,
            duration_seconds=1,
            metadata={},
        )
        mock_game_res2 = GameResult(
            game_id="g2e",
            agent_info=mock_agent_info,
            opponent_info=opp2,
            winner=1,
            moves_count=1,
            duration_seconds=1,
            metadata={},
        )

        evaluator._play_games_against_opponent = AsyncMock(
            side_effect=[([mock_game_res1], []), ([mock_game_res2], [])]
        )

        mock_standings = {"standings_data": "some_data"}
        evaluator._calculate_tournament_standings = MagicMock(
            return_value=mock_standings
        )

        mock_summary_stats_instance = MagicMock(spec=SummaryStats)
        mock_summary_stats_instance.total_games = 2

        with patch(
            "keisei.evaluation.core.SummaryStats.from_games",
            MagicMock(return_value=mock_summary_stats_instance),
        ) as mock_from_games:
            result = await evaluator.evaluate(mock_agent_info, mock_evaluation_context)

        assert evaluator._load_tournament_opponents.call_count == 1
        assert evaluator._play_games_against_opponent.call_count == 2
        evaluator._play_games_against_opponent.assert_any_call(
            mock_agent_info, opp1, 1, mock_evaluation_context
        )
        evaluator._play_games_against_opponent.assert_any_call(
            mock_agent_info, opp2, 1, mock_evaluation_context
        )

        mock_from_games.assert_called_once_with([mock_game_res1, mock_game_res2])
        evaluator._calculate_tournament_standings.assert_called_once_with(
            [mock_game_res1, mock_game_res2], mock_opponents_list, mock_agent_info
        )

        assert result.games == [mock_game_res1, mock_game_res2]
        assert result.summary_stats == mock_summary_stats_instance
        assert result.analytics_data["tournament_specific_analytics"] == mock_standings
        assert result.errors == []
        evaluator.log_evaluation_start.assert_called_once()
        evaluator.log_evaluation_complete.assert_called_once_with(result)

    # --- Tests for Game Playing Helper Methods ---

    @pytest.mark.asyncio
    async def test_game_load_evaluation_entity_agent(
        self, mock_tournament_config, mock_agent_info, mock_evaluation_context
    ):
        evaluator = TournamentEvaluator(mock_tournament_config)
        with patch(
            "keisei.evaluation.strategies.tournament.load_evaluation_agent",
            new_callable=AsyncMock,
        ) as mock_load_agent:
            mock_loaded_agent = MagicMock()
            mock_load_agent.return_value = mock_loaded_agent

            loaded_entity = await evaluator._game_load_evaluation_entity(
                mock_agent_info, "cpu", 46
            )
            assert loaded_entity == mock_loaded_agent
            mock_load_agent.assert_called_once_with(
                checkpoint_path=mock_agent_info.checkpoint_path,
                device_str="cpu",
                policy_mapper=evaluator.policy_mapper,
                input_channels=46,
            )

    @pytest.mark.asyncio
    async def test_game_load_evaluation_entity_opponent_ppo(
        self, mock_tournament_config, mock_evaluation_context
    ):
        evaluator = TournamentEvaluator(mock_tournament_config)
        mock_ppo_opponent_info = OpponentInfo(
            name="PPOOpp", type="ppo_agent", checkpoint_path="/path/to/ppo_opp.ptk"
        )

        with patch(
            "keisei.evaluation.strategies.tournament.load_evaluation_agent",
            new_callable=AsyncMock,
        ) as mock_load_agent:
            mock_loaded_ppo_opp = MagicMock()
            mock_load_agent.return_value = mock_loaded_ppo_opp

            loaded_entity = await evaluator._game_load_evaluation_entity(
                mock_ppo_opponent_info, "cuda", 46
            )
            assert loaded_entity == mock_loaded_ppo_opp
            mock_load_agent.assert_called_once_with(
                checkpoint_path=mock_ppo_opponent_info.checkpoint_path,
                device_str="cuda",
                policy_mapper=evaluator.policy_mapper,
                input_channels=46,
            )

    @pytest.mark.asyncio
    async def test_game_load_evaluation_entity_opponent_other(
        self, mock_tournament_config, mock_evaluation_context
    ):
        evaluator = TournamentEvaluator(mock_tournament_config)
        mock_heuristic_opponent_info = OpponentInfo(
            name="HeuristicOpp", type="heuristic_v1", checkpoint_path="/path/h.dll"
        )

        with patch(
            "keisei.evaluation.strategies.tournament.initialize_opponent",
            new_callable=AsyncMock,
        ) as mock_init_opp:
            mock_initialized_opp = MagicMock()
            mock_init_opp.return_value = mock_initialized_opp

            loaded_entity = await evaluator._game_load_evaluation_entity(
                mock_heuristic_opponent_info, "cpu", 46
            )
            assert loaded_entity == mock_initialized_opp
            mock_init_opp.assert_called_once_with(
                opponent_type=mock_heuristic_opponent_info.type,
                opponent_path=mock_heuristic_opponent_info.checkpoint_path,
                device_str="cpu",
                policy_mapper=evaluator.policy_mapper,
                input_channels=46,
            )

    @pytest.mark.asyncio
    async def test_game_load_evaluation_entity_unknown_type(
        self, mock_tournament_config, mock_evaluation_context
    ):
        evaluator = TournamentEvaluator(mock_tournament_config)
        mock_unknown_info = MagicMock()

        with pytest.raises(ValueError, match="Unknown entity type for loading"):
            await evaluator._game_load_evaluation_entity(mock_unknown_info, "cpu", 46)

    @pytest.mark.asyncio
    async def test_game_get_player_action_ppo_agent(self, mock_tournament_config):
        evaluator = TournamentEvaluator(mock_tournament_config)
        mock_player_entity = MagicMock()
        mock_player_entity.select_action = MagicMock(
            return_value=("move_data", "action_info")
        )
        mock_game = MagicMock(spec=ShogiGame)
        mock_game.get_observation = MagicMock(return_value="observation_data")
        mock_legal_mask = MagicMock()

        action = await evaluator._game_get_player_action(
            mock_player_entity, mock_game, mock_legal_mask
        )

        assert action == "move_data"
        mock_player_entity.select_action.assert_called_once_with(
            "observation_data", mock_legal_mask, is_training=False
        )

    @pytest.mark.asyncio
    async def test_game_get_player_action_heuristic_opponent(
        self, mock_tournament_config
    ):
        evaluator = TournamentEvaluator(mock_tournament_config)
        mock_player_entity = MagicMock()
        # Make sure select_action is not present or is None for this mock, to test the elif path
        mock_player_entity.select_action = None
        mock_player_entity.select_move = MagicMock(return_value="heuristic_move")
        mock_game = MagicMock(spec=ShogiGame)
        mock_legal_mask = MagicMock()

        action = await evaluator._game_get_player_action(
            mock_player_entity, mock_game, mock_legal_mask
        )

        assert action == "heuristic_move"
        mock_player_entity.select_move.assert_called_once_with(mock_game)

    @pytest.mark.asyncio
    async def test_game_get_player_action_unsupported_entity(
        self, mock_tournament_config
    ):
        evaluator = TournamentEvaluator(mock_tournament_config)
        mock_player_entity = MagicMock()
        mock_player_entity.select_action = None
        mock_player_entity.select_move = None
        mock_game = MagicMock(spec=ShogiGame)
        mock_legal_mask = MagicMock()

        with pytest.raises(TypeError, match="Unsupported player entity type"):
            await evaluator._game_get_player_action(
                mock_player_entity, mock_game, mock_legal_mask
            )

    @pytest.mark.asyncio
    async def test_game_validate_and_make_move_valid(self, mock_tournament_config):
        evaluator = TournamentEvaluator(mock_tournament_config)
        mock_game = MagicMock(spec=ShogiGame)
        mock_game.make_move = MagicMock()
        mock_game.game_over = False

        valid_move = "7g7f"
        legal_moves = ["7g7f", "2c2d"]

        result = await evaluator._game_validate_and_make_move(
            mock_game, valid_move, legal_moves, 0, "TestPlayer"
        )

        assert result is True
        mock_game.make_move.assert_called_once_with(valid_move)
        assert mock_game.game_over is False

    @pytest.mark.asyncio
    async def test_game_validate_and_make_move_none_move(self, mock_tournament_config):
        evaluator = TournamentEvaluator(mock_tournament_config)
        mock_game = MagicMock(spec=ShogiGame)
        mock_game.game_over = False

        result = await evaluator._game_validate_and_make_move(
            mock_game, None, ["7g7f"], 0, "TestPlayer"
        )

        assert result is False
        assert mock_game.game_over is True
        assert mock_game.winner == Color(1)
        assert mock_game.termination_reason == TERMINATION_REASON_ILLEGAL_MOVE

    @pytest.mark.asyncio
    async def test_game_validate_and_make_move_illegal_move(
        self, mock_tournament_config
    ):
        evaluator = TournamentEvaluator(mock_tournament_config)
        mock_game = MagicMock(spec=ShogiGame)
        mock_game.game_over = False

        illegal_move = "1a1b"
        legal_moves = ["7g7f", "2c2d"]

        result = await evaluator._game_validate_and_make_move(
            mock_game, illegal_move, legal_moves, 1, "TestPlayer"
        )

        assert result is False
        assert mock_game.game_over is True
        assert mock_game.winner == Color(0)
        assert mock_game.termination_reason == TERMINATION_REASON_ILLEGAL_MOVE

    @pytest.mark.asyncio
    async def test_game_validate_and_make_move_execution_error(
        self, mock_tournament_config
    ):
        evaluator = TournamentEvaluator(mock_tournament_config)
        mock_game = MagicMock(spec=ShogiGame)
        mock_game.make_move = MagicMock(
            side_effect=RuntimeError("ShogiGame internal error")
        )
        mock_game.game_over = False

        valid_move = "7g7f"
        legal_moves = ["7g7f"]

        result = await evaluator._game_validate_and_make_move(
            mock_game, valid_move, legal_moves, 0, "TestPlayer"
        )

        assert result is False
        assert mock_game.game_over is True
        assert mock_game.winner == Color(1)
        assert (
            mock_game.termination_reason
            == f"{TERMINATION_REASON_MOVE_EXECUTION_ERROR}: ShogiGame internal error"
        )

    @pytest.mark.asyncio
    async def test_handle_no_legal_moves_shogigame_sets_winner(
        self, mock_tournament_config
    ):
        evaluator = TournamentEvaluator(mock_tournament_config)
        mock_game = MagicMock(spec=ShogiGame)
        mock_game.current_player = Color(0)  # Sente
        mock_game.winner = Color(1)  # Gote wins (e.g. checkmate)
        mock_game.termination_reason = "Checkmate"

        await evaluator._handle_no_legal_moves(mock_game)

        assert mock_game.game_over is True
        assert mock_game.winner == Color(1)
        assert mock_game.termination_reason == "Checkmate"

    @pytest.mark.asyncio
    async def test_handle_no_legal_moves_shogigame_no_winner_set(
        self, mock_tournament_config
    ):
        evaluator = TournamentEvaluator(mock_tournament_config)
        mock_game = MagicMock(spec=ShogiGame)
        mock_game.current_player = Color(0)  # Sente
        mock_game.winner = None
        mock_game.termination_reason = None

        with patch(
            "keisei.evaluation.strategies.tournament.logger", MagicMock()
        ) as mock_logger:
            await evaluator._handle_no_legal_moves(mock_game)

        assert mock_game.game_over is True
        assert mock_game.winner is None
        assert (
            mock_game.termination_reason
            == TERMINATION_REASON_NO_LEGAL_MOVES_UNDETERMINED
        )
        mock_logger.warning.assert_called_once()

    @pytest.mark.asyncio
    async def test_game_process_one_turn_successful(
        self, mock_tournament_config, mock_evaluation_context
    ):
        evaluator = TournamentEvaluator(mock_tournament_config)
        mock_game = MagicMock(spec=ShogiGame)
        mock_game.get_legal_moves = MagicMock(return_value=["7g7f"])
        mock_game.current_player = Color(0)  # Sente

        mock_player_entity = MagicMock()
        mock_player_entity.device = torch.device("cpu")

        evaluator._game_get_player_action = AsyncMock(return_value="7g7f")
        evaluator._game_validate_and_make_move = AsyncMock(return_value=True)
        evaluator.policy_mapper.get_legal_mask = MagicMock(
            return_value="legal_mask_tensor"
        )

        result = await evaluator._game_process_one_turn(
            mock_game, mock_player_entity, mock_evaluation_context
        )

        assert result is True
        mock_game.get_legal_moves.assert_called_once()
        evaluator.policy_mapper.get_legal_mask.assert_called_once_with(
            ["7g7f"], torch.device("cpu")
        )
        evaluator._game_get_player_action.assert_called_once_with(
            mock_player_entity, mock_game, "legal_mask_tensor"
        )
        evaluator._game_validate_and_make_move.assert_called_once_with(
            mock_game, "7g7f", ["7g7f"], 0, type(mock_player_entity).__name__
        )

    @pytest.mark.asyncio
    async def test_game_process_one_turn_no_legal_moves(
        self, mock_tournament_config, mock_evaluation_context
    ):
        evaluator = TournamentEvaluator(mock_tournament_config)
        mock_game = MagicMock(spec=ShogiGame)
        mock_game.get_legal_moves = MagicMock(return_value=[])

        mock_player_entity = MagicMock()
        evaluator._handle_no_legal_moves = AsyncMock()

        result = await evaluator._game_process_one_turn(
            mock_game, mock_player_entity, mock_evaluation_context
        )

        assert result is False
        evaluator._handle_no_legal_moves.assert_called_once_with(mock_game)

    @pytest.mark.asyncio
    async def test_game_process_one_turn_action_selection_error(
        self, mock_tournament_config, mock_evaluation_context
    ):
        evaluator = TournamentEvaluator(mock_tournament_config)
        mock_game = MagicMock(spec=ShogiGame)
        mock_game.get_legal_moves = MagicMock(return_value=["7g7f"])
        mock_game.current_player = Color(0)  # Sente
        mock_game.game_over = False

        mock_player_entity = MagicMock()
        mock_player_entity.device = torch.device("cpu")
        evaluator._game_get_player_action = AsyncMock(
            side_effect=RuntimeError("Action selection failed")
        )
        evaluator.policy_mapper.get_legal_mask = MagicMock(
            return_value="legal_mask_tensor"
        )

        result = await evaluator._game_process_one_turn(
            mock_game, mock_player_entity, mock_evaluation_context
        )

        assert result is False
        assert mock_game.game_over is True
        assert mock_game.winner == Color(1)
        assert (
            mock_game.termination_reason
            == f"{TERMINATION_REASON_ACTION_SELECTION_ERROR}: Action selection failed"
        )

    @pytest.mark.asyncio
    async def test_game_process_one_turn_invalid_move(
        self, mock_tournament_config, mock_evaluation_context
    ):
        evaluator = TournamentEvaluator(mock_tournament_config)
        mock_game = MagicMock(spec=ShogiGame)
        mock_game.get_legal_moves = MagicMock(return_value=["7g7f"])
        mock_game.current_player = Color(0)  # Sente

        mock_player_entity = MagicMock()
        mock_player_entity.device = torch.device("cpu")
        evaluator._game_get_player_action = AsyncMock(return_value="illegal_move")
        evaluator._game_validate_and_make_move = AsyncMock(return_value=False)
        evaluator.policy_mapper.get_legal_mask = MagicMock(
            return_value="legal_mask_tensor"
        )

        result = await evaluator._game_process_one_turn(
            mock_game, mock_player_entity, mock_evaluation_context
        )

        assert result is False
        evaluator._game_validate_and_make_move.assert_called_once()

    # --- Tests for _game_process_one_turn ---

    @patch(
        "keisei.evaluation.strategies.tournament.TournamentEvaluator._game_get_player_action"
    )
    async def test_game_process_one_turn_valid_move_agent_turn(self, mock_get_action, mock_tournament_config, mock_evaluation_context):
        """Tests _game_process_one_turn: agent's turn, valid move made."""
        evaluator = TournamentEvaluator(mock_tournament_config)
        mock_game = MagicMock(spec=ShogiGame)
        mock_game.get_legal_moves.return_value = ["7g7f"]
        mock_game.current_player = Color(0)  # Sente
        mock_game.game_over = False
        
        mock_player_entity = MagicMock()
        mock_player_entity.device = torch.device("cpu")
        
        mock_get_action.return_value = "7g7f"
        evaluator._game_validate_and_make_move = AsyncMock(return_value=True)
        evaluator.policy_mapper.get_legal_mask = MagicMock(return_value="legal_mask_tensor")

        result = await evaluator._game_process_one_turn(
            mock_game, mock_player_entity, mock_evaluation_context
        )

        assert result is True
        mock_get_action.assert_called_once_with(
            mock_player_entity, mock_game, "legal_mask_tensor"
        )
        evaluator._game_validate_and_make_move.assert_called_once()

    @patch(
        "keisei.evaluation.strategies.tournament.TournamentEvaluator._game_get_player_action"
    )
    async def test_game_process_one_turn_valid_move_opponent_turn(
        self, mock_get_action, mock_tournament_config, mock_evaluation_context
    ):
        """Tests _game_process_one_turn: opponent's turn, valid move made."""
        evaluator = TournamentEvaluator(mock_tournament_config)
        mock_game = MagicMock(spec=ShogiGame)
        mock_game.get_legal_moves.return_value = ["2g2f"]
        mock_game.current_player = Color(1)  # Gote
        mock_game.game_over = False
        
        mock_player_entity = MagicMock()
        mock_player_entity.device = torch.device("cpu")
        
        mock_get_action.return_value = "2g2f"
        evaluator._game_validate_and_make_move = AsyncMock(return_value=True)
        evaluator.policy_mapper.get_legal_mask = MagicMock(return_value="legal_mask_tensor")

        result = await evaluator._game_process_one_turn(
            mock_game, mock_player_entity, mock_evaluation_context
        )

        assert result is True
        mock_get_action.assert_called_once_with(
            mock_player_entity, mock_game, "legal_mask_tensor"
        )
        evaluator._game_validate_and_make_move.assert_called_once()

    @patch(
        "keisei.evaluation.strategies.tournament.TournamentEvaluator._game_get_player_action"
    )
    async def test_game_process_one_turn_invalid_move(self, mock_get_action, mock_tournament_config, mock_evaluation_context):
        """Tests _game_process_one_turn: player makes an invalid move."""
        evaluator = TournamentEvaluator(mock_tournament_config)
        mock_game = MagicMock(spec=ShogiGame)
        mock_game.get_legal_moves.return_value = ["7g7f"]
        mock_game.current_player = Color(0)  # Sente
        mock_game.game_over = False
        
        mock_player_entity = MagicMock()
        mock_player_entity.device = torch.device("cpu")
        
        mock_get_action.return_value = "1a1b"  # Invalid move
        evaluator._game_validate_and_make_move = AsyncMock(return_value=False)  # Validation fails
        evaluator.policy_mapper.get_legal_mask = MagicMock(return_value="legal_mask_tensor")

        result = await evaluator._game_process_one_turn(
            mock_game, mock_player_entity, mock_evaluation_context
        )

        assert result is False
        mock_get_action.assert_called_once_with(
            mock_player_entity, mock_game, "legal_mask_tensor"
        )
        evaluator._game_validate_and_make_move.assert_called_once()

    @patch(
        "keisei.evaluation.strategies.tournament.TournamentEvaluator._game_get_player_action"
    )
    async def test_game_process_one_turn_no_action_no_legal_moves_stalemate(
        self, mock_get_action, mock_tournament_config, mock_evaluation_context
    ):
        """Tests _game_process_one_turn: no action from policy, no legal moves (stalemate)."""
        evaluator = TournamentEvaluator(mock_tournament_config)
        mock_game = MagicMock(spec=ShogiGame)
        mock_game.get_legal_moves.return_value = []  # No legal moves
        mock_game.current_player = Color(0)  # Sente
        mock_game.game_over = False
        
        mock_player_entity = MagicMock()
        mock_player_entity.device = torch.device("cpu")
        
        evaluator._handle_no_legal_moves = AsyncMock()

        result = await evaluator._game_process_one_turn(
            mock_game, mock_player_entity, mock_evaluation_context
        )

        assert result is False
        evaluator._handle_no_legal_moves.assert_called_once_with(mock_game)
        mock_get_action.assert_not_called()

    @patch(
        "keisei.evaluation.strategies.tournament.TournamentEvaluator._game_get_player_action"
    )
    async def test_game_process_one_turn_no_action_with_legal_moves_policy_error(
        self, mock_get_action, mock_tournament_config, mock_evaluation_context
    ):
        """Tests _game_process_one_turn: action selection error."""
        evaluator = TournamentEvaluator(mock_tournament_config)
        mock_game = MagicMock(spec=ShogiGame)
        mock_game.get_legal_moves.return_value = ["7g7f"]
        mock_game.current_player = Color(0)  # Sente
        mock_game.game_over = False
        
        mock_player_entity = MagicMock()
        mock_player_entity.device = torch.device("cpu")
        
        mock_get_action.side_effect = RuntimeError("Action selection failed")
        evaluator.policy_mapper.get_legal_mask = MagicMock(return_value="legal_mask_tensor")

        result = await evaluator._game_process_one_turn(
            mock_game, mock_player_entity, mock_evaluation_context
        )

        assert result is False
        assert mock_game.game_over is True
        assert mock_game.winner == Color(1)  # Other player wins
        assert "Action selection error: Action selection failed" in mock_game.termination_reason

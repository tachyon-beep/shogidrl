"""Tournament opponent management and entity loading tests.

This module contains tests for loading tournament opponents from configuration
and loading different types of evaluation entities (agents and opponents).
"""
import pytest
from unittest.mock import MagicMock, patch

from keisei.evaluation.core import OpponentInfo
from keisei.evaluation.strategies.tournament import TournamentEvaluator


class TestTournamentOpponents:
    """Test tournament opponent management functionality."""

    @pytest.mark.asyncio
    async def test_load_tournament_opponents_empty_config(
        self, mock_tournament_config, mock_evaluation_context
    ):
        """Test loading opponents with empty configuration."""
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
        """Test loading opponents with valid configuration."""
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
        """Test loading opponents when some are already OpponentInfo objects."""
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
        """Test loading opponents with malformed configuration entry."""
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
        """Test loading opponents when processing an entry causes an error."""
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
            # Verify error was logged (not warning, as exceptions should log errors)
            mock_logger.error.assert_called()


class TestEntityLoading:
    """Test evaluation entity loading functionality."""

    def test_game_load_evaluation_entity_agent(
        self, mock_tournament_config, mock_agent_info, mock_evaluation_context
    ):
        """Test loading agent entity."""
        evaluator = TournamentEvaluator(mock_tournament_config)
        with patch(
            "keisei.evaluation.strategies.tournament.load_evaluation_agent",
            new_callable=MagicMock,
        ) as mock_load_agent:
            mock_loaded_agent = MagicMock()
            mock_load_agent.return_value = mock_loaded_agent

            loaded_entity = evaluator._load_evaluation_entity(
                mock_agent_info, "cpu", 46
            )
            assert loaded_entity == mock_loaded_agent
            mock_load_agent.assert_called_once_with(
                checkpoint_path=mock_agent_info.checkpoint_path,
                device_str="cpu",
                policy_mapper=evaluator.policy_mapper,
                input_channels=46
            )

    def test_game_load_evaluation_entity_opponent_ppo(
        self, mock_tournament_config, mock_evaluation_context
    ):
        """Test loading PPO opponent entity."""
        evaluator = TournamentEvaluator(mock_tournament_config)
        mock_ppo_opponent_info = OpponentInfo(
            name="PPOOpp", type="ppo_agent", checkpoint_path="/path/to/ppo_opp.ptk"
        )

        with patch(
            "keisei.evaluation.strategies.tournament.load_evaluation_agent",
            new_callable=MagicMock,
        ) as mock_load_agent:
            mock_loaded_ppo_opp = MagicMock()
            mock_load_agent.return_value = mock_loaded_ppo_opp

            loaded_entity = evaluator._load_evaluation_entity(
                mock_ppo_opponent_info, "cpu", 46
            )
            assert loaded_entity == mock_loaded_ppo_opp
            mock_load_agent.assert_called_once_with(
                checkpoint_path="/path/to/ppo_opp.ptk",
                device_str="cpu",
                policy_mapper=evaluator.policy_mapper,
                input_channels=46
            )

    def test_game_load_evaluation_entity_opponent_other(
        self, mock_tournament_config, mock_evaluation_context
    ):
        """Test loading non-PPO opponent entity."""
        evaluator = TournamentEvaluator(mock_tournament_config)
        mock_heuristic_opponent_info = OpponentInfo(
            name="HeuristicOpp", type="heuristic_v1", checkpoint_path="/path/h.dll"
        )

        with patch(
            "keisei.evaluation.strategies.tournament.initialize_opponent",
            new_callable=MagicMock,
        ) as mock_init_opp:
            mock_initialized_opp = MagicMock()
            mock_init_opp.return_value = mock_initialized_opp

            loaded_entity = evaluator._load_evaluation_entity(
                mock_heuristic_opponent_info, "cpu", 46
            )
            assert loaded_entity == mock_initialized_opp
            mock_init_opp.assert_called_once_with(
                opponent_type="heuristic_v1",
                opponent_path="/path/h.dll",
                device_str="cpu",
                policy_mapper=evaluator.policy_mapper,
                input_channels=46
            )

    def test_load_evaluation_entity_unknown_type(
        self, mock_tournament_config, mock_evaluation_context
    ):
        """Test loading entity with unknown type raises error."""
        evaluator = TournamentEvaluator(mock_tournament_config)
        mock_unknown_info = MagicMock()

        with pytest.raises(ValueError, match="Unknown entity type for loading"):
            evaluator._load_evaluation_entity(mock_unknown_info, "cpu", 46)

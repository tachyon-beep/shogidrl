#!/usr/bin/env python3
"""Debug script to understand the logger test expectations."""

import sys
import os
sys.path.insert(0, '/home/john/keisei')

import asyncio
from unittest.mock import MagicMock, patch
from keisei.evaluation.strategies.tournament import TournamentEvaluator
from keisei.evaluation.core import OpponentInfo

async def debug_logger_test():
    print("Starting debug test...")
    mock_tournament_config = MagicMock()
    opponent_data_1 = {"name": "Opp1", "type": "random"}
    problematic_data = {
        "name": "ProblemOpp", 
        "type": "special_type_that_breaks_init",
    }
    mock_tournament_config.opponent_pool_config = [
        opponent_data_1,
        problematic_data,
    ]
    
    print("Creating evaluator...")
    evaluator = TournamentEvaluator(mock_tournament_config)
    
    with patch("keisei.evaluation.strategies.tournament.logger", MagicMock()) as mock_logger:
        print("Patching OpponentInfo...")
        original_opponent_info_init = OpponentInfo.__init__
        
        def mock_init(self_obj, *args, **kwargs_init):
            print(f"mock_init called with name: {kwargs_init.get('name')}")
            if kwargs_init.get("name") == "ProblemOpp":
                print("Raising ValueError for ProblemOpp")
                raise ValueError("Simulated init error")
            print("Calling original init")
            original_opponent_info_init(self_obj, *args, **kwargs_init)
        
        with patch("keisei.evaluation.core.OpponentInfo.__init__", mock_init):
            print("Loading tournament opponents...")
            try:
                opponents = await evaluator._load_tournament_opponents()
                print(f"Got {len(opponents)} opponents")
            except Exception as e:
                print(f"Error loading opponents: {e}")
        
        print("Logger error call count:", mock_logger.error.call_count)
        if mock_logger.error.call_count > 0:
            args, kwargs = mock_logger.error.call_args
            print("Call args:", args)
            print("Call kwargs:", kwargs)
            print("Args length:", len(args))
            for i, arg in enumerate(args):
                print(f"args[{i}]: {repr(arg)}")
            
            # Test what the test is checking
            print("\nTest checks:")
            print(f"'Failed to load opponent from config data at index 1' in args[0]: {'Failed to load opponent from config data at index 1' in args[0]}")
            if len(args) > 2:
                print(f"'Simulated init error' in str(args[2]): {'Simulated init error' in str(args[2])}")
            else:
                print("args[2] does not exist")

if __name__ == "__main__":
    asyncio.run(debug_logger_test())

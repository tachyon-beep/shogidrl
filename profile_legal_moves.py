'''Profiling script for generate_all_legal_moves.'''
import cProfile
import pstats
import io
import traceback # Added for error logging

from keisei.shogi.shogi_game import ShogiGame
from keisei.shogi.shogi_rules_logic import generate_all_legal_moves

def profile_generate_moves():
    """Sets up a game and profiles generate_all_legal_moves."""
    try: # Added try block
        game = ShogiGame()
        # Optional: Set up a more complex board state if needed via SFEN
        # game.sfen_board_setup("sfen_string_here") 

        profiler = cProfile.Profile()
        print("Profiler initialized.") # Added print statement
        profiler.enable()
        print("Profiler enabled.") # Added print statement

        # Run the function to be profiled
        print("Profiling generate_all_legal_moves...") # Added print statement
        _ = generate_all_legal_moves(game) 
        print("Profiling finished.") # Added print statement

        profiler.disable()
        print("Profiler disabled.") # Added print statement

        # Save and print stats
        profile_file = "legal_moves.prof"
        profiler.dump_stats(profile_file)
        print(f"Profiling data dumped to {profile_file}") # Added print statement

        print("\nProfiling results for generate_all_legal_moves (top 30 by cumtime):")
        s = io.StringIO()
        # Sort by cumulative time, and print top 30
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumtime') 
        ps.print_stats(30)
        output_stats = s.getvalue()
        if output_stats:
            print(output_stats)
        else:
            print("No stats to print (output was empty).") # Added for clarity
        
        # Also print to a file for good measure
        with open("legal_moves_stats.txt", "w") as f_out:
            f_out.write("Profiling results for generate_all_legal_moves (top 30 by cumtime):\n")
            # Re-create pstats to write to file if s was already used or closed
            ps_file = pstats.Stats(profiler, stream=f_out).sort_stats('cumtime')
            ps_file.print_stats(30)
        print("Detailed stats also written to legal_moves_stats.txt")

    except Exception as e: # Added except block
        print("An error occurred during profiling:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {e}")
        print("Traceback:")
        traceback.print_exc()

if __name__ == "__main__":
    profile_generate_moves()

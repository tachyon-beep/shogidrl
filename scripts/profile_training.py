#!/usr/bin/env python3
"""
Performance profiling script for Keisei Shogi training.
Implements Task 4.1 from the remediation plan.
"""

import argparse
import cProfile
import os
import pstats
import sys
import tempfile
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from keisei.training.trainer import Trainer
from keisei.utils import load_config


def profile_training_run(total_timesteps=2048, output_file=None):
    """
    Run a profiled training session to identify performance bottlenecks.

    Args:
        total_timesteps: Number of timesteps to run for profiling
        output_file: Output file for profiling results (default: auto-generated)
    """
    if output_file is None:
        output_file = f"keisei_profile_{total_timesteps}steps.prof"

    print(f"Starting profiled training run for {total_timesteps} timesteps...")
    print(f"Profile output will be saved to: {output_file}")

    # Setup configuration for profiling
    config = load_config()

    # Override settings for profiling run
    config.training.total_timesteps = total_timesteps
    config.training.checkpoint_interval_timesteps = (
        total_timesteps // 2
    )  # One checkpoint
    # Skip periodic evaluation by setting a very large interval
    config.training.evaluation_interval_timesteps = total_timesteps * 10
    config.wandb.enabled = False  # Disable W&B for clean profiling
    config.demo.enable_demo_mode = False  # Disable demo mode

    # Create temporary directory for this run
    with tempfile.TemporaryDirectory() as temp_dir:
        config.logging.model_dir = temp_dir

        def run_training():
            """Training function to be profiled."""
            # Create a simple args object for the Trainer
            class ProfileArgs:
                def __init__(self):
                    self.run_name = "profile_run"
                    self.resume = None
            
            args = ProfileArgs()
            trainer = Trainer(config, args)
            trainer.run_training_loop()

        # Run with profiler
        cProfile.run("run_training()", output_file)

    print(f"Profiling complete. Results saved to: {output_file}")
    return output_file


def analyze_profile(profile_file, top_n=20):
    """
    Analyze and display profiling results.

    Args:
        profile_file: Path to the .prof file
        top_n: Number of top functions to display
    """
    print(f"\nAnalyzing profile: {profile_file}")
    print("=" * 60)

    # Load profiling stats
    stats = pstats.Stats(profile_file)

    # Sort by cumulative time and display top functions
    print(f"\nTop {top_n} functions by cumulative time:")
    print("-" * 60)
    stats.sort_stats("cumulative").print_stats(top_n)

    # Sort by internal time and display top functions
    print(f"\nTop {top_n} functions by internal time:")
    print("-" * 60)
    stats.sort_stats("time").print_stats(top_n)

    # Look for specific Shogi engine functions
    print(f"\nShogi engine functions (move generation, validation):")
    print("-" * 60)
    stats.sort_stats("cumulative").print_stats(".*shogi.*")

    # Look for experience buffer and agent functions
    print(f"\nRL-specific functions (agent, buffer, learning):")
    print("-" * 60)
    stats.sort_stats("cumulative").print_stats(".*agent.*|.*buffer.*|.*learn.*")


def generate_profile_report(profile_file, report_file=None):
    """
    Generate a detailed profiling report.

    Args:
        profile_file: Path to the .prof file
        report_file: Output file for the report (default: auto-generated)
    """
    if report_file is None:
        report_file = profile_file.replace(".prof", "_report.txt")

    print(f"Generating detailed report: {report_file}")

    stats = pstats.Stats(profile_file)

    with open(report_file, "w") as f:
        # Write header
        f.write("Keisei Shogi Performance Profiling Report\n")
        f.write("=" * 50 + "\n\n")

        # Redirect stats output to file
        old_stdout = sys.stdout
        sys.stdout = f

        try:
            # Overall statistics
            f.write("OVERALL STATISTICS:\n")
            f.write("-" * 30 + "\n")
            stats.print_stats()

            f.write("\n\nTOP FUNCTIONS BY CUMULATIVE TIME:\n")
            f.write("-" * 40 + "\n")
            stats.sort_stats("cumulative").print_stats(30)

            f.write("\n\nTOP FUNCTIONS BY INTERNAL TIME:\n")
            f.write("-" * 40 + "\n")
            stats.sort_stats("time").print_stats(30)

            f.write("\n\nSHOGI ENGINE FUNCTIONS:\n")
            f.write("-" * 30 + "\n")
            stats.sort_stats("cumulative").print_stats(".*shogi.*")

            f.write("\n\nRL AGENT FUNCTIONS:\n")
            f.write("-" * 25 + "\n")
            stats.sort_stats("cumulative").print_stats(".*agent.*|.*buffer.*|.*learn.*")

        finally:
            sys.stdout = old_stdout

    print(f"Report saved to: {report_file}")
    return report_file


def main():
    """Main entry point for the profiling script."""
    parser = argparse.ArgumentParser(
        description="Profile Keisei Shogi training performance"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=2048,
        help="Number of timesteps to run for profiling (default: 2048)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file for profiling results (default: auto-generated)",
    )
    parser.add_argument(
        "--analyze-only",
        type=str,
        help="Skip profiling and only analyze existing profile file",
    )
    parser.add_argument(
        "--report", action="store_true", help="Generate a detailed text report"
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of top functions to display in analysis (default: 20)",
    )

    args = parser.parse_args()

    if args.analyze_only:
        # Only analyze existing profile
        profile_file = args.analyze_only
        if not os.path.exists(profile_file):
            print(f"Error: Profile file not found: {profile_file}")
            return 1
    else:
        # Run profiling
        try:
            profile_file = profile_training_run(
                total_timesteps=args.timesteps, output_file=args.output
            )
        except Exception as e:
            print(f"Error during profiling: {e}")
            return 1

    # Analyze results
    try:
        analyze_profile(profile_file, top_n=args.top_n)

        if args.report:
            generate_profile_report(profile_file)

    except Exception as e:
        print(f"Error during analysis: {e}")
        return 1

    # Provide suggestions for using snakeviz
    print(f"\nTo view interactive flame graph, install snakeviz and run:")
    print(f"  pip install snakeviz")
    print(f"  snakeviz {profile_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

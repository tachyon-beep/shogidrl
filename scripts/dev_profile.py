#!/usr/bin/env python3
"""
Development profiling helper for interactive optimization.
Provides quick profiling tools for development workflow.
"""

import argparse
import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.profile_training import profile_training_run, analyze_profile, generate_profile_report


def quick_profile(timesteps=1024, analyze=True):
    """
    Quick profiling run for development.
    
    Args:
        timesteps: Number of timesteps to profile (default: 1024 for speed)
        analyze: Whether to immediately analyze results
        
    Returns:
        Path to profile file
    """
    print(f"🔍 Running quick development profile ({timesteps} timesteps)...")
    
    output_file = f"dev_profile_{timesteps}steps.prof"
    profile_file = profile_training_run(timesteps, output_file)
    
    if analyze and os.path.exists(profile_file):
        print("\n📊 Analyzing profile results...")
        analyze_profile(profile_file, top_n=10)
        
        # Generate brief report
        report_file = profile_file.replace(".prof", "_brief.txt")
        generate_profile_report(profile_file, report_file)
        print(f"📄 Brief report saved to: {report_file}")
    
    return profile_file


def compare_profiles(baseline_file, current_file):
    """
    Compare two profiling runs for regression detection.
    
    Args:
        baseline_file: Path to baseline profile
        current_file: Path to current profile
    """
    import pstats
    
    if not os.path.exists(baseline_file):
        print(f"❌ Baseline file not found: {baseline_file}")
        return False
        
    if not os.path.exists(current_file):
        print(f"❌ Current file not found: {current_file}")
        return False
    
    print(f"🔄 Comparing profiles:")
    print(f"  Baseline: {baseline_file}")
    print(f"  Current:  {current_file}")
    
    # Load both profiles
    baseline_stats = pstats.Stats(baseline_file)
    current_stats = pstats.Stats(current_file)
    
    # Get total time for each
    baseline_total = baseline_stats.total_tt
    current_total = current_stats.total_tt
    
    # Calculate performance difference
    if baseline_total > 0:
        improvement = ((baseline_total - current_total) / baseline_total) * 100
        if improvement > 0:
            print(f"✅ Performance improved by {improvement:.1f}%")
        elif improvement < -5:  # More than 5% regression
            print(f"⚠️  Performance regressed by {abs(improvement):.1f}%")
        else:
            print(f"📊 Performance similar (±{abs(improvement):.1f}%)")
    else:
        print("⚠️  Could not compare - baseline total time is 0")
    
    print(f"\nBaseline total time: {baseline_total:.3f}s")
    print(f"Current total time:  {current_total:.3f}s")
    
    return True


def setup_baseline():
    """Set up a baseline profile for future comparisons."""
    print("🎯 Creating baseline profile...")
    baseline_file = quick_profile(timesteps=2048, analyze=False)
    
    # Move to standard baseline location
    baseline_path = "baseline_profile.prof"
    if os.path.exists(baseline_file):
        os.rename(baseline_file, baseline_path)
        print(f"✅ Baseline profile saved as: {baseline_path}")
        return baseline_path
    
    return None


def main():
    """Main entry point for development profiling."""
    parser = argparse.ArgumentParser(
        description="Development profiling helper for Keisei Shogi"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Quick profile command
    quick_parser = subparsers.add_parser("quick", help="Run quick development profile")
    quick_parser.add_argument(
        "--timesteps", type=int, default=1024,
        help="Number of timesteps to profile (default: 1024)"
    )
    quick_parser.add_argument(
        "--no-analyze", action="store_true",
        help="Skip immediate analysis"
    )
    
    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare two profiles")
    compare_parser.add_argument("baseline", help="Baseline profile file")
    compare_parser.add_argument("current", help="Current profile file")
    
    # Baseline setup command
    subparsers.add_parser("baseline", help="Create baseline profile")
    
    # Full profile command
    full_parser = subparsers.add_parser("full", help="Run full profile with analysis")
    full_parser.add_argument(
        "--timesteps", type=int, default=2048,
        help="Number of timesteps to profile (default: 2048)"
    )
    
    args = parser.parse_args()
    
    if args.command == "quick":
        profile_file = quick_profile(args.timesteps, not args.no_analyze)
        print(f"\n🎯 Profile complete: {profile_file}")
        
    elif args.command == "compare":
        compare_profiles(args.baseline, args.current)
        
    elif args.command == "baseline":
        baseline_file = setup_baseline()
        if baseline_file:
            print(f"\n🎯 Use 'python scripts/dev_profile.py compare {baseline_file} <new_profile>' for comparisons")
            
    elif args.command == "full":
        print(f"🔍 Running full profile with {args.timesteps} timesteps...")
        profile_file = profile_training_run(args.timesteps)
        analyze_profile(profile_file)
        generate_profile_report(profile_file)
        print(f"\n🎯 Full profile complete: {profile_file}")
        
    else:
        parser.print_help()
        print("\nCommon usage:")
        print("  python scripts/dev_profile.py quick           # Quick development profile")
        print("  python scripts/dev_profile.py baseline        # Set up baseline")
        print("  python scripts/dev_profile.py full            # Full detailed profile")


if __name__ == "__main__":
    main()

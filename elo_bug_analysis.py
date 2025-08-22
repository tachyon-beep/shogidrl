#!/usr/bin/env python3
"""
Comprehensive ELO Tracker Bug Analysis Script
"""

import sys
import traceback
from typing import Optional, Dict, Any
import math

# Test imports
try:
    from keisei.evaluation.analytics.elo_tracker import EloTracker
    print("✓ EloTracker import successful")
except ImportError as e:
    print(f"✗ EloTracker import failed: {e}")
    sys.exit(1)

# Test basic functionality
def test_basic_functionality():
    """Test basic ELO tracker functionality"""
    print("\n=== BASIC FUNCTIONALITY TESTS ===")
    
    try:
        tracker = EloTracker()
        print("✓ EloTracker instantiation successful")
    except Exception as e:
        print(f"✗ EloTracker instantiation failed: {e}")
        return False
    
    # Test rating retrieval for new player
    try:
        rating = tracker.get_rating("new_player")
        expected = tracker.default_initial_rating
        assert rating == expected, f"Expected {expected}, got {rating}"
        print(f"✓ New player rating initialization: {rating}")
    except Exception as e:
        print(f"✗ New player rating failed: {e}")
        return False
    
    # Test rating update
    try:
        old_a = tracker.get_rating("player_a")
        old_b = tracker.get_rating("player_b")
        new_a, new_b = tracker.update_rating("player_a", "player_b", 1.0)
        
        # Basic validation
        assert new_a > old_a, f"Winner should gain rating: {old_a} -> {new_a}"
        assert new_b < old_b, f"Loser should lose rating: {old_b} -> {new_b}"
        print(f"✓ Rating update: A {old_a} -> {new_a}, B {old_b} -> {new_b}")
    except Exception as e:
        print(f"✗ Rating update failed: {e}")
        traceback.print_exc()
        return False
    
    return True

def test_mathematical_correctness():
    """Test mathematical correctness of ELO calculations"""
    print("\n=== MATHEMATICAL CORRECTNESS TESTS ===")
    
    tracker = EloTracker()
    
    # Test expected score formula
    try:
        rating_a, rating_b = 1600, 1400
        expected_a = tracker._expected_score(rating_a, rating_b)
        
        # Manual calculation
        manual_expected = 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))
        
        assert abs(expected_a - manual_expected) < 1e-10, f"Expected score mismatch: {expected_a} vs {manual_expected}"
        print(f"✓ Expected score calculation: {expected_a:.6f}")
    except Exception as e:
        print(f"✗ Expected score calculation failed: {e}")
        return False
    
    # Test rating conservation
    try:
        tracker.add_entity("conservation_a", 1500)
        tracker.add_entity("conservation_b", 1500)
        
        initial_sum = tracker.get_rating("conservation_a") + tracker.get_rating("conservation_b")
        new_a, new_b = tracker.update_rating("conservation_a", "conservation_b", 0.5)  # Draw
        final_sum = new_a + new_b
        
        assert abs(initial_sum - final_sum) < 1e-10, f"Rating conservation violated: {initial_sum} != {final_sum}"
        print(f"✓ Rating conservation maintained: {initial_sum} = {final_sum}")
    except Exception as e:
        print(f"✗ Rating conservation test failed: {e}")
        return False
    
    # Test symmetry
    try:
        rating_x, rating_y = 1700, 1300
        expected_x = tracker._expected_score(rating_x, rating_y)
        expected_y = tracker._expected_score(rating_y, rating_x)
        
        sum_expected = expected_x + expected_y
        assert abs(sum_expected - 1.0) < 1e-10, f"Expected scores don't sum to 1: {sum_expected}"
        print(f"✓ Expected score symmetry: {expected_x:.6f} + {expected_y:.6f} = {sum_expected:.6f}")
    except Exception as e:
        print(f"✗ Expected score symmetry test failed: {e}")
        return False
    
    return True

def test_edge_cases():
    """Test edge cases and error conditions"""
    print("\n=== EDGE CASE TESTS ===")
    
    tracker = EloTracker()
    
    # Test invalid score values
    invalid_scores = [-0.1, 1.1, -1.0, 2.0, float('inf'), -float('inf')]
    for score in invalid_scores:
        try:
            tracker.update_rating("test_a", "test_b", score)
            print(f"⚠ Invalid score {score} was accepted (potential bug)")
        except Exception:
            # Expected to fail, but silently accepting is also a bug
            pass
    
    # Test NaN score
    try:
        tracker.update_rating("test_a", "test_b", float('nan'))
        print("⚠ NaN score was accepted (potential bug)")
    except Exception:
        pass
    
    # Test same player ID
    try:
        tracker.update_rating("same", "same", 1.0)
        print("⚠ Self-play was accepted (may be intentional or bug)")
    except Exception:
        pass
    
    # Test extreme ratings
    try:
        tracker.add_entity("extreme_low", -10000)
        tracker.add_entity("extreme_high", 10000)
        
        expected = tracker._expected_score(-10000, 10000)
        if math.isnan(expected) or math.isinf(expected):
            print(f"✗ Extreme ratings cause mathematical errors: {expected}")
            return False
        
        print(f"✓ Extreme ratings handled: {expected:.10f}")
    except Exception as e:
        print(f"✗ Extreme ratings test failed: {e}")
        return False
    
    # Test zero K-factor
    try:
        zero_k_tracker = EloTracker(default_k_factor=0)
        old_rating = zero_k_tracker.get_rating("zero_k_test")
        new_rating, _ = zero_k_tracker.update_rating("zero_k_test", "opponent", 1.0)
        
        assert old_rating == new_rating, f"Zero K-factor should not change ratings: {old_rating} vs {new_rating}"
        print("✓ Zero K-factor handled correctly")
    except Exception as e:
        print(f"✗ Zero K-factor test failed: {e}")
        return False
    
    return True

def test_data_integrity():
    """Test data management and persistence"""
    print("\n=== DATA INTEGRITY TESTS ===")
    
    tracker = EloTracker()
    
    # Test history tracking
    try:
        initial_history_length = len(tracker.get_rating_history())
        
        tracker.update_rating("history_a", "history_b", 1.0)
        tracker.update_rating("history_b", "history_c", 0.5)
        
        final_history_length = len(tracker.get_rating_history())
        expected_length = initial_history_length + 2
        
        assert final_history_length == expected_length, f"History tracking failed: expected {expected_length}, got {final_history_length}"
        print(f"✓ History tracking: {initial_history_length} -> {final_history_length}")
    except Exception as e:
        print(f"✗ History tracking failed: {e}")
        return False
    
    # Test data consistency after multiple updates
    try:
        ratings_before = tracker.get_all_ratings().copy()
        
        # Perform multiple updates
        for i in range(100):
            player_a = f"consistency_a_{i % 5}"
            player_b = f"consistency_b_{i % 5}"
            score = [0.0, 0.5, 1.0][i % 3]
            tracker.update_rating(player_a, player_b, score)
        
        ratings_after = tracker.get_all_ratings()
        
        # Check for data corruption
        corrupted_ratings = []
        for player_id, rating in ratings_after.items():
            if not isinstance(rating, (int, float)):
                corrupted_ratings.append((player_id, rating, type(rating)))
            elif math.isnan(rating) or math.isinf(rating):
                corrupted_ratings.append((player_id, rating, "non-finite"))
        
        if corrupted_ratings:
            print(f"✗ Data corruption detected: {corrupted_ratings}")
            return False
        
        print(f"✓ Data integrity maintained after {len(ratings_after)} ratings")
    except Exception as e:
        print(f"✗ Data integrity test failed: {e}")
        return False
    
    return True

def test_integration_compatibility():
    """Test integration with evaluation system"""
    print("\n=== INTEGRATION COMPATIBILITY TESTS ===")
    
    try:
        from keisei.evaluation.core.evaluation_result import EvaluationResult
        print("✓ EvaluationResult import successful")
    except ImportError as e:
        print(f"✗ EvaluationResult import failed: {e}")
        return False
    
    # Test ELO snapshot functionality
    try:
        tracker = EloTracker()
        tracker.add_entity("integration_test", 1600)
        
        snapshot = tracker.get_all_ratings()
        assert isinstance(snapshot, dict), f"Snapshot should be dict, got {type(snapshot)}"
        assert "integration_test" in snapshot, "Snapshot should contain added entity"
        assert snapshot["integration_test"] == 1600, "Snapshot should have correct rating"
        
        print("✓ ELO snapshot functionality works")
    except Exception as e:
        print(f"✗ ELO snapshot test failed: {e}")
        return False
    
    # Test leaderboard functionality
    try:
        leaderboard = tracker.get_leaderboard(top_n=5)
        assert isinstance(leaderboard, list), f"Leaderboard should be list, got {type(leaderboard)}"
        
        if len(leaderboard) > 1:
            # Check sorting
            for i in range(len(leaderboard) - 1):
                assert leaderboard[i][1] >= leaderboard[i+1][1], "Leaderboard not sorted by rating"
        
        print(f"✓ Leaderboard functionality: {len(leaderboard)} entries")
    except Exception as e:
        print(f"✗ Leaderboard test failed: {e}")
        return False
    
    return True

def main():
    """Run all bug analysis tests"""
    print("ELO TRACKER BUG ANALYSIS")
    print("=" * 50)
    
    test_results = []
    
    # Run all test suites
    test_suites = [
        ("Basic Functionality", test_basic_functionality),
        ("Mathematical Correctness", test_mathematical_correctness),
        ("Edge Cases", test_edge_cases),
        ("Data Integrity", test_data_integrity),
        ("Integration Compatibility", test_integration_compatibility),
    ]
    
    for suite_name, test_func in test_suites:
        try:
            result = test_func()
            test_results.append((suite_name, result))
        except Exception as e:
            print(f"\n✗ {suite_name} test suite crashed: {e}")
            traceback.print_exc()
            test_results.append((suite_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("BUG ANALYSIS SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, result in test_results if result)
    total = len(test_results)
    
    for suite_name, result in test_results:
        status = "PASS" if result else "FAIL"
        print(f"{suite_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} test suites passed")
    
    if passed < total:
        print("\n⚠ BUGS DETECTED - See test output above for details")
        return 1
    else:
        print("\n✓ No critical bugs detected in basic functionality")
        return 0

if __name__ == "__main__":
    sys.exit(main())
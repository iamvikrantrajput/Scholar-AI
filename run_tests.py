#!/usr/bin/env python3
"""
Test runner for AI Research Assistant
Runs all test files in the tests/ directory
"""

import sys
import subprocess
from pathlib import Path

def run_test(test_file):
    """Run a single test file and return success status."""
    print(f"\n{'='*60}")
    print(f"🧪 Running: {test_file.name}")
    print('='*60)
    
    try:
        result = subprocess.run([
            sys.executable, str(test_file)
        ], capture_output=True, text=True, timeout=120)
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
            
        success = result.returncode == 0
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"\n{status}: {test_file.name}")
        
        return success
        
    except subprocess.TimeoutExpired:
        print(f"⏰ TIMEOUT: {test_file.name}")
        return False
    except Exception as e:
        print(f"❌ ERROR: {test_file.name} - {e}")
        return False

def main():
    """Run all tests in the tests directory."""
    print("🔬 AI Research Assistant - Test Suite")
    print("="*60)
    
    tests_dir = Path("tests")
    if not tests_dir.exists():
        print("❌ Tests directory not found!")
        return False
        
    test_files = list(tests_dir.glob("test_*.py"))
    if not test_files:
        print("❌ No test files found in tests/")
        return False
        
    print(f"📋 Found {len(test_files)} test files")
    
    # Run all tests
    results = []
    for test_file in sorted(test_files):
        success = run_test(test_file)
        results.append((test_file.name, success))
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 TEST SUMMARY")
    print('='*60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅" if success else "❌"
        print(f"  {status} {test_name}")
    
    print(f"\n🎯 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED!")
        return True
    else:
        print(f"⚠️  {total - passed} tests failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
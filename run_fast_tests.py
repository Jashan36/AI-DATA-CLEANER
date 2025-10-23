#!/usr/bin/env python3
"""
Fast test runner - skips slow and web tests
"""
import subprocess
import sys


def run_fast_tests():
    """Run only fast tests."""
    cmd = [
        sys.executable, "-m", "pytest",
        "-q",
        "--tb=short",
        "-x",  # Stop on first failure
        "-W", "ignore",
        "-m", "not slow and not web",
        "tests/"
    ]

    print("🚀 Running fast tests only...")
    print(" ".join(cmd))
    return subprocess.call(cmd)


def run_specific_tests(test_files):
    """Run specific test files quickly."""
    cmd = [
        sys.executable, "-m", "pytest",
        "-q",
        "--tb=short",
        "-x",
        "-W", "ignore"
    ] + test_files

    print("🚀 Running specific tests...")
    return subprocess.call(cmd)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Run specific tests
        sys.exit(run_specific_tests(sys.argv[1:]))
    else:
        # Run all fast tests
        result = run_fast_tests()
        sys.exit(result)


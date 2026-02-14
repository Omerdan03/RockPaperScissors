"""Run all e2e tests.

Usage:
    python tests/run.py              # headless (default)
    python tests/run.py --headed     # visible browser window
"""

import argparse
import sys
import os

parser = argparse.ArgumentParser(description="Run RPS Stratego e2e tests")
parser.add_argument(
    "--headed", action="store_true",
    help="Show the browser window instead of running headless",
)
args, remaining = parser.parse_known_args()

test_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(test_dir, ".."))

pytest_args = [test_dir, "-v"] + remaining
if args.headed:
    pytest_args.append("--headed")

sys.exit(__import__("pytest").main(pytest_args))

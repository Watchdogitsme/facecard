"""Convenience runner for test_system."""
import subprocess
import sys

if __name__ == "__main__":
    exit_code = subprocess.call([sys.executable, "test_system.py"])
    sys.exit(exit_code)

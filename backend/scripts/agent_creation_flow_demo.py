#!/usr/bin/env python3
import os
import sys

# Ensure backend root is on path
CURRENT_DIR = os.path.dirname(__file__)
BACKEND_ROOT = os.path.dirname(CURRENT_DIR)
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)


def main():
    from test_agent_creation_flow import main as _main  # type: ignore
    _main()


if __name__ == "__main__":
    main()


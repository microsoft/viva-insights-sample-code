#!/usr/bin/env python3
"""Compatibility entry point for deterministic prompt-page and catalogue checks."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from generate_prompt_catalogue import main

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "--check"]
    sys.exit(main())

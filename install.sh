#!/bin/bash

set -e

# Create virtual environment
uv venv --python 3.13 --clear

# Activate virtual environment
source .venv/bin/activate

# Install using uv.lock (deterministic, fast, modern)
uv sync --all-extras

# Install pre-commit hooks
pre-commit install
pre-commit autoupdate

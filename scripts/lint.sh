#!/bin/bash
set -e

echo "Running black..."
poetry run black --check .

echo "Running isort..."
poetry run isort --check .

echo "Running pyright..."
poetry run pyright

echo "All checks passed!"

#!/usr/bin/env bash
set -euo pipefail

# Repo root = directory containing this script
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default target if no args (after options) are passed
DEFAULT_TARGET="${ROOT_DIR}/python/CuTeDSL/cutlass/cnc_dsl"

FIX=0
TARGET_DIRS=()

# Parse options
while [[ $# -gt 0 ]]; do
  case "$1" in
    --fix|-f)
      FIX=1
      shift
      ;;
    *)
      TARGET_DIRS+=("$1")
      shift
      ;;
  esac
done

# If no explicit targets, use the default
if [ "${#TARGET_DIRS[@]}" -eq 0 ]; then
  TARGET_DIRS=("$DEFAULT_TARGET")
fi

echo "Linting with ruff..."

ruff format "${TARGET_DIRS[@]}"
if [ "$FIX" -eq 1 ]; then
  ruff check --fix "${TARGET_DIRS[@]}"
else
  ruff check "${TARGET_DIRS[@]}"
fi

echo "Type checking with mypy..."
mypy --config-file mypy.ini "${TARGET_DIRS[@]}"
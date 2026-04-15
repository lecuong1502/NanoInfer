#!/usr/bin/env bash
# tools/clean.sh — deletes all build artifacts
#
# Usage:
# bash tools/clean.sh # deletes build/ and .so
# bash tools/clean.sh --all # deletes reports/ and __pycache__ as well

set -euo pipefail
 
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
 
CLEAN_ALL=0
if [[ "${1:-}" == "--all" ]]; then
    CLEAN_ALL=1
fi
 
echo "Cleaning build artifacts in: $ROOT_DIR"
 
# CMake build directory
if [[ -d "build" ]]; then
    rm -rf build
    echo "  Removed: build/"
fi
 
# Python extension .so files in nanoinfer/ package
find nanoinfer/ -name "*.so" -delete 2>/dev/null && echo "  Removed: nanoinfer/*.so" || true

 Python egg-info (pip install -e . artifacts)
find . -maxdepth 2 -name "*.egg-info" -type d -exec rm -rf {} + 2>/dev/null || true
echo "  Removed: *.egg-info"
 
# Python cache
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true
echo "  Removed: __pycache__ and .pyc"
 
if [[ "$CLEAN_ALL" == "1" ]]; then
    # Profiling reports
    if [[ -d "reports" ]]; then
        rm -rf reports
        echo "  Removed: reports/"
    fi
 
    # CUDA compilation cache
    if [[ -d "${HOME}/.nv" ]]; then
        echo "  [skip] ~/.nv/ (CUDA cache) — remove manually if needed"
    fi
fi

echo "Done!"
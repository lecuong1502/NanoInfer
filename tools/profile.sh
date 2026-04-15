#!/usr/bin/env bash
# tools/profile.sh — NVIDIA Nsight Systems profiling wrapper
#
# Run a binary or Python script under Nsight Systems and automatically
# open the report, or print the path to open it manually.
#
# Usage:
#   bash tools/profile.sh ./build/benchmarks/bench_gemm
#   bash tools/profile.sh ./build/benchmarks/bench_attention
#   bash tools/profile.sh python benchmarks/bench_e2e.py --precision fp16
#
# Output: reports/profile_<tên_binary>_<timestamp>.nsys-rep
#
# Requirements:
#   nsys (NVIDIA Nsight Systems)
#   or install private from https://developer.nvidia.com/nsight-systems
#
# Key metrics to look at in the Nsight GUI:
#   - CUDA HW: kernel timeline, occupancy, memory throughput
#   - CUDA API: cudaMalloc/cudaMemcpy overhead
#   - GPU metrics: SM utilization, DRAM bandwidth (compared to peak 2TB/s on the A100)

set -euo pipefail

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
REPORTS_DIR="reports"
NSYS_BIN="${NSYS_BIN:-nsys}"          # override: NSYS_BIN=/path/to/nsys bash profile.sh ...
OPEN_GUI="${OPEN_GUI:-0}"             # set OPEN_GUI=1 to open Nsight GUI after profile

# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------
if [[ $# -eq 0 ]]; then
    echo "Usage: bash tools/profile.sh <binary_or_script> [args...]"
    echo ""
    echo "Examples:"
    echo "  bash tools/profile.sh ./build/benchmarks/bench_gemm"
    echo "  bash tools/profile.sh ./build/benchmarks/bench_attention"
    echo "  bash tools/profile.sh python benchmarks/bench_e2e.py --precision fp16"
    exit 1
fi

if ! command -v "$NSYS_BIN" &>/dev/null; then
    echo "[error] nsys not found. Install Nsight Systems or set NSYS_BIN=/path/to/nsys"
    echo "        Download: https://developer.nvidia.com/nsight-systems"
    exit 1
fi

# ---------------------------------------------------------------------------
# Build output filename from binary name + timestamp
# ---------------------------------------------------------------------------
mkdir -p "$REPORTS_DIR"
 
BINARY_NAME=$(basename "$1" | sed 's/\.[^.]*$//')  # strip extension
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
REPORT_NAME="${REPORTS_DIR}/profile_${BINARY_NAME}_${TIMESTAMP}"
 
# ---------------------------------------------------------------------------
# Run Nsight Systems
# ---------------------------------------------------------------------------
echo "========================================"
echo "Profiling: $*"
echo "Output:    ${REPORT_NAME}.nsys-rep"
echo "========================================"
echo ""

"$NSYS_BIN" profile \
    --output="$REPORT_NAME" \
    --trace=cuda,nvtx,osrt \
    --cuda-memory-usage=true \
    --force-overwrite=true \
    --stats=true \
    "$@"
 
echo ""
echo "========================================"
echo "Profile complete: ${REPORT_NAME}.nsys-rep"
echo ""
echo "Key stats printed above. To view full timeline:"
echo "  nsys-ui ${REPORT_NAME}.nsys-rep"
echo ""

# ---------------------------------------------------------------------------
# Print quick summary from CLI stats
# ---------------------------------------------------------------------------
echo "[Quick summary — top 10 CUDA kernels by time]"
"$NSYS_BIN" stats \
    --report gputrace \
    --format table \
    --output - \
    "${REPORT_NAME}.nsys-rep" 2>/dev/null | head -20 || true

echo ""
echo "[Memory transfers]"
"$NSYS_BIN" stats \
    --report cudaapisum \
    --format table \
    --output - \
    "${REPORT_NAME}.nsys-rep" 2>/dev/null | grep -E "Memcpy|Memset|total" | head -10 || true
 
echo "========================================"

# Open GUI if requested
if [[ "${OPEN_GUI}" == "1" ]]; then
    if command -v nsys-ui &>/dev/null; then
        nsys-ui "${REPORT_NAME}.nsys-rep" &
    else
        echo "[info] nsys-ui not found — open the .nsys-rep file manually"
    fi
fi
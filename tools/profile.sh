#!/usr/bin/env bash
# tools/profile.sh — NVIDIA Nsight Systems profiling wrapper
#
# Chạy một binary hoặc Python script dưới Nsight Systems,
# in stats ra terminal, và hướng dẫn cách đọc file .nsys-rep.
#
# Cách dùng:
#   bash tools/profile.sh ./build/benchmarks/bench_gemm
#   bash tools/profile.sh ./build/benchmarks/bench_attention
#   bash tools/profile.sh python benchmarks/bench_e2e.py --precision fp16
#
# Output: reports/<binary>_<timestamp>.nsys-rep
#
# Đọc file .nsys-rep:
#   GUI:  nsys-ui reports/<file>.nsys-rep
#         (hoặc download Nsight Systems GUI từ https://developer.nvidia.com/nsight-systems
#          copy file .nsys-rep về máy local nếu đang SSH)
#   CLI:  nsys stats --report cuda_gpu_kern_sum   reports/<file>.nsys-rep
#         nsys stats --report cuda_gpu_mem_time_sum reports/<file>.nsys-rep
#         nsys stats --report cuda_api_sum          reports/<file>.nsys-rep
#   CSV:  nsys stats --report cuda_gpu_kern_sum --format csv --output reports/out reports/<file>.nsys-rep
#
# Yêu cầu:
#   nsys >= 2022.1 — thường nằm ở /usr/local/cuda/bin/nsys hoặc /opt/nvidia/nsight-systems/*/bin/nsys
#   Override path: NSYS_BIN=/path/to/nsys bash tools/profile.sh ...

set -euo pipefail

# ---------------------------------------------------------------------------
# Config — override bằng environment variables nếu cần
# ---------------------------------------------------------------------------
REPORTS_DIR="${REPORTS_DIR:-reports}"
NSYS_BIN="${NSYS_BIN:-nsys}"
OPEN_GUI="${OPEN_GUI:-0}"   # OPEN_GUI=1 bash tools/profile.sh ... để tự mở GUI

# ---------------------------------------------------------------------------
# Validate args
# ---------------------------------------------------------------------------
if [[ $# -eq 0 ]]; then
    echo "Usage: bash tools/profile.sh <binary> [args...]"
    echo ""
    echo "Examples:"
    echo "  bash tools/profile.sh ./build/benchmarks/bench_gemm"
    echo "  bash tools/profile.sh ./build/benchmarks/bench_softmax"
    echo "  bash tools/profile.sh ./build/benchmarks/bench_attention"
    echo "  bash tools/profile.sh python benchmarks/bench_e2e.py --precision fp16"
    exit 1
fi

if ! command -v "$NSYS_BIN" &>/dev/null; then
    echo "[error] nsys not found."
    echo "        Add CUDA bin to PATH: export PATH=/usr/local/cuda/bin:\$PATH"
    echo "        Or set: NSYS_BIN=/path/to/nsys bash tools/profile.sh ..."
    echo "        Download: https://developer.nvidia.com/nsight-systems"
    exit 1
fi

# ---------------------------------------------------------------------------
# Detect nsys version — một số flags chỉ có từ version nhất định
# ---------------------------------------------------------------------------
NSYS_VERSION=$("$NSYS_BIN" --version 2>&1 | grep -oP '\d+\.\d+' | head -1 || echo "0.0")
NSYS_MAJOR=$(echo "$NSYS_VERSION" | cut -d. -f1)

# --cuda-graph-trace=node có từ nsys 2021.x trở lên
GRAPH_TRACE_FLAG=""
if [[ "$NSYS_MAJOR" -ge 2021 ]]; then
    GRAPH_TRACE_FLAG="--cuda-graph-trace=node"
fi

# ---------------------------------------------------------------------------
# Output path
# ---------------------------------------------------------------------------
mkdir -p "$REPORTS_DIR"

BINARY_NAME=$(basename "$1" | sed 's/\.[^.]*$//')
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
REPORT_BASE="${REPORTS_DIR}/${BINARY_NAME}_${TIMESTAMP}"

# ---------------------------------------------------------------------------
# Run nsys profile
# ---------------------------------------------------------------------------
echo "========================================"
echo "  Binary:  $*"
echo "  Output:  ${REPORT_BASE}.nsys-rep"
echo "  nsys:    $("$NSYS_BIN" --version 2>&1 | head -1)"
echo "========================================"
echo ""

"$NSYS_BIN" profile \
    --output="$REPORT_BASE" \
    --trace=cuda,nvtx \
    --cpuctxsw=none \
    ${GRAPH_TRACE_FLAG:+"$GRAPH_TRACE_FLAG"} \
    --cuda-memory-usage=true \
    --force-overwrite=true \
    --stats=true \
    "$@"

echo ""
echo "========================================"
echo "Profile complete: ${REPORT_BASE}.nsys-rep"
echo "========================================"

# ---------------------------------------------------------------------------
# Print key stats từ file vừa tạo
# ---------------------------------------------------------------------------
echo ""
echo "[1] CUDA kernels — sorted by GPU time"
"$NSYS_BIN" stats \
    --report cuda_gpu_kern_sum \
    --format table \
    --output - \
    "${REPORT_BASE}.nsys-rep" 2>/dev/null || echo "  (no kernel data)"

echo ""
echo "[2] Memory transfers — H↔D"
"$NSYS_BIN" stats \
    --report cuda_gpu_mem_time_sum \
    --format table \
    --output - \
    "${REPORT_BASE}.nsys-rep" 2>/dev/null || echo "  (no memory transfer data)"

echo ""
echo "[3] CUDA API overhead"
"$NSYS_BIN" stats \
    --report cuda_api_sum \
    --format table \
    --output - \
    "${REPORT_BASE}.nsys-rep" 2>/dev/null | head -20 || echo "  (no API data)"

# ---------------------------------------------------------------------------
# Export CSV cho Python/Excel (optional)
# ---------------------------------------------------------------------------
"$NSYS_BIN" stats \
    --report cuda_gpu_kern_sum \
    --format csv \
    --output "${REPORT_BASE}_kernels" \
    "${REPORT_BASE}.nsys-rep" 2>/dev/null && \
    echo "" && \
    echo "CSV exported: ${REPORT_BASE}_kernels.csv" || true

# ---------------------------------------------------------------------------
# Hướng dẫn đọc file
# ---------------------------------------------------------------------------
echo ""
echo "========================================"
echo "Cách đọc file .nsys-rep:"
echo ""
echo "  CLI (không cần GUI):"
echo "    nsys stats --report cuda_gpu_kern_sum    ${REPORT_BASE}.nsys-rep"
echo "    nsys stats --report cuda_gpu_mem_time_sum ${REPORT_BASE}.nsys-rep"
echo "    nsys stats --report cuda_api_sum          ${REPORT_BASE}.nsys-rep"
echo ""
echo "  GUI (visual timeline):"
if command -v nsys-ui &>/dev/null; then
    echo "    nsys-ui ${REPORT_BASE}.nsys-rep"
else
    echo "    nsys-ui không có trên máy này."
    echo "    → Copy file về máy local rồi mở bằng Nsight Systems GUI:"
    echo "      scp $(hostname):$(pwd)/${REPORT_BASE}.nsys-rep ."
    echo "      Download GUI: https://developer.nvidia.com/nsight-systems"
fi
echo "========================================"

# ---------------------------------------------------------------------------
# Mở GUI nếu OPEN_GUI=1
# ---------------------------------------------------------------------------
if [[ "${OPEN_GUI}" == "1" ]]; then
    if command -v nsys-ui &>/dev/null; then
        echo ""
        echo "Opening nsys-ui..."
        nsys-ui "${REPORT_BASE}.nsys-rep" &
    else
        echo ""
        echo "[info] OPEN_GUI=1 nhưng nsys-ui không tìm thấy."
    fi
fi
#!/usr/bin/env bash
# tools/gpu_info.sh — Print GPU information and check the build environment
#
# Usage:
#   bash tools/gpu_info.sh
#
# Output:
#   - GPU model, compute capability, VRAM
#   - CUDA toolkit version
#   - nvcc path and version
#   - cuBLAS, cuDNN version (if available)
#   - CUDA_ARCHITECTURES suggestion for CMake

set -euo pipefail
 
SEP="=================================================="
 
echo "$SEP"
echo "NanoInfer — GPU & CUDA environment check"
echo "$SEP"
 
# ---------------------------------------------------------------------------
# 1. GPU info via nvidia-smi
# ---------------------------------------------------------------------------
echo ""
echo "[GPU]"

if ! command -v nvidia-smi &>/dev/null; then
    echo "  [error] nvidia-smi not found — is the NVIDIA driver installed?"
    exit 1
fi
 
nvidia-smi --query-gpu=index,name,driver_version,memory.total,compute_cap \
           --format=csv,noheader | while IFS=, read -r idx name drv mem cc; do
    # Trim whitespace
    idx=$(echo "$idx" | xargs)
    name=$(echo "$name" | xargs)
    drv=$(echo "$drv" | xargs)
    mem=$(echo "$mem" | xargs)
    cc=$(echo "$cc" | xargs)
 
    echo "  GPU $idx: $name"
    echo "    Driver:          $drv"
    echo "    VRAM:            $mem"
    echo "    Compute cap:     $cc"
 
    # Map compute capability to architecture name and CMake flag
    case "$cc" in
        7.0) arch="Volta (V100)"   ; sm="70" ;;
        7.5) arch="Turing (T4)"   ; sm="75" ;;
        8.0) arch="Ampere (A100)" ; sm="80" ;;
        8.6) arch="Ampere (RTX 3090/A6000)"; sm="86" ;;
        8.9) arch="Ada (RTX 4090)"; sm="89" ;;
        9.0) arch="Hopper (H100)" ; sm="90" ;;
        *)   arch="Unknown"        ; sm="??"  ;;
    esac
 
    echo "    Architecture:    $arch"
    echo "    CMake flag:      -DCUDA_ARCHITECTURES=$sm"
 
    # FP8 support check (requires sm_90+)
    if [[ "$sm" == "90" ]]; then
        echo "    FP8 support:     YES (cuda_fp8.h available)"
    else
        echo "    FP8 support:     NO  (requires Hopper sm_90)"
    fi
done

# ---------------------------------------------------------------------------
# 2. CUDA toolkit
# ---------------------------------------------------------------------------
echo ""
echo "[CUDA toolkit]"
 
if command -v nvcc &>/dev/null; then
    NVCC_PATH=$(which nvcc)
    NVCC_VER=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
    CUDA_HOME_DETECTED=$(dirname "$(dirname "$NVCC_PATH")")
 
    echo "  nvcc:         $NVCC_PATH"
    echo "  CUDA version: $NVCC_VER"
    echo "  CUDA_HOME:    $CUDA_HOME_DETECTED"
 
    # Check include path layout (standard vs targets/)
    STD_INC="$CUDA_HOME_DETECTED/include/cuda_runtime.h"
    TGT_INC="$CUDA_HOME_DETECTED/targets/x86_64-linux/include/cuda_runtime.h"
    if [[ -f "$STD_INC" ]]; then
        echo "  Include path: $CUDA_HOME_DETECTED/include  (standard layout)"
    elif [[ -f "$TGT_INC" ]]; then
        echo "  Include path: $CUDA_HOME_DETECTED/targets/x86_64-linux/include  (non-standard layout)"
        echo "  [note] setup.py auto-detects this — no manual fix needed"
    else
        echo "  [warn] cuda_runtime.h not found — check CUDA installation"
    fi
else
    echo "  [warn] nvcc not found — add CUDA bin dir to PATH:"
    echo "         export PATH=/usr/local/cuda/bin:\$PATH"
fi
 
# ---------------------------------------------------------------------------
# 3. cuBLAS
# ---------------------------------------------------------------------------
echo ""
echo "[cuBLAS]"
CUBLAS_H=$(find /usr/local/cuda* /usr/cuda 2>/dev/null \
           -name "cublas_v2.h" 2>/dev/null | head -1 || true)
if [[ -n "$CUBLAS_H" ]]; then
    VER=$(grep "CUBLAS_VER_MAJOR\|CUBLAS_VER_MINOR\|CUBLAS_VER_PATCH" \
          "$(dirname "$CUBLAS_H")/cublas_api.h" 2>/dev/null \
          | awk '{print $3}' | tr '\n' '.' | sed 's/\.$//' || echo "unknown")
    echo "  Found: $CUBLAS_H"
    echo "  Version: $VER"
else
    echo "  [warn] cublas_v2.h not found"
fi
 
# ---------------------------------------------------------------------------
# 4. Python environment
# ---------------------------------------------------------------------------
echo ""
echo "[Python]"
if command -v python3 &>/dev/null; then
    PY_VER=$(python3 --version)
    echo "  $PY_VER ($(which python3))"
 
    # pybind11
    if python3 -c "import pybind11; print('  pybind11:', pybind11.__version__, 'at', pybind11.get_include())" 2>/dev/null; then
        true
    else
        echo "  [warn] pybind11 not found — install: pip install pybind11"
    fi
 
    # PyTorch (for test reference)
    if python3 -c "import torch; print('  torch:', torch.__version__, '| CUDA:', torch.version.cuda)" 2>/dev/null; then
        true
    else
        echo "  [info] torch not found — needed for correctness tests"
        echo "         install: pip install torch"
    fi
else
    echo "  [warn] python3 not found"
fi
 
# ---------------------------------------------------------------------------
# 5. Build tools
# ---------------------------------------------------------------------------
echo ""
echo "[Build tools]"
 
for tool in cmake make g++; do
    if command -v "$tool" &>/dev/null; then
        VER=$("$tool" --version 2>&1 | head -1)
        echo "  $tool: $VER"
    else
        echo "  [warn] $tool not found"
    fi
done
 
# CMake version check (need >= 3.20)
if command -v cmake &>/dev/null; then
    CMAKE_VER=$(cmake --version | head -1 | awk '{print $3}')
    MAJOR=$(echo "$CMAKE_VER" | cut -d. -f1)
    MINOR=$(echo "$CMAKE_VER" | cut -d. -f2)
    if [[ "$MAJOR" -lt 3 || ( "$MAJOR" -eq 3 && "$MINOR" -lt 20 ) ]]; then
        echo "  [warn] CMake $CMAKE_VER < 3.20 required — please upgrade"
    fi
fi

echo ""
echo "$SEP"
echo "Run 'bash tools/profile.sh ./build/benchmarks/bench_gemm' to profile"
echo "$SEP"
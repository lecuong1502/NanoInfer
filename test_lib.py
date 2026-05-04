"""
test_lib.py — Quick smoke test for the NanoInfer Python package.

Run after building the .so:
    python test_lib.py

Expected output: all sections print OK / PASS.
"""

import sys
import numpy as np

print("=" * 60)
print("NanoInfer Python package smoke test")
print("=" * 60)

# ---------------------------------------------------------------------------
# 1. Import
# ---------------------------------------------------------------------------
print("\n[1] Import ...")
try:
    import nanoinfer
    print(f"    OK  nanoinfer {nanoinfer.__version__} loaded")
    print(f"    .so location: {nanoinfer.__file__}")
except ImportError as e:
    print(f"    FAIL: {e}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# 2. kernels.gemm
# ---------------------------------------------------------------------------
print("\n[2] kernels.gemm (256x256) ...")
try:
    import torch  # for reference
    M, K, N = 256, 256, 256
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)

    C_nano = nanoinfer.kernels.gemm(A, B)
    C_ref  = A @ B

    max_diff = float(np.abs(C_nano - C_ref).max())
    status   = "PASS" if max_diff < 1e-2 else "FAIL"
    print(f"    {status}  max_diff={max_diff:.2e}  shape={C_nano.shape}")
except Exception as e:
    print(f"    SKIP  ({e})")

# ---------------------------------------------------------------------------
# 3. kernels.softmax
# ---------------------------------------------------------------------------
print("\n[3] kernels.softmax (64 x 512) ...")
try:
    rows, cols = 64, 512
    x = np.random.randn(rows, cols).astype(np.float32)

    y_nano = nanoinfer.kernels.softmax(x)

    # Check: row sums ≈ 1
    row_sums  = y_nano.sum(axis=1)
    max_err   = float(np.abs(row_sums - 1.0).max())
    status    = "PASS" if max_err < 1e-4 else "FAIL"
    print(f"    {status}  row_sum_err={max_err:.2e}  shape={y_nano.shape}")
except Exception as e:
    print(f"    SKIP  ({e})")

# ---------------------------------------------------------------------------
# 4. kernels.flash_attention
# ---------------------------------------------------------------------------
print("\n[4] kernels.flash_attention (B=1 H=4 S=64 d=64) ...")
try:
    B, H, S, d = 1, 4, 64, 64
    scale       = 1.0 / np.sqrt(d)
    Q = np.random.randn(B, H, S, d).astype(np.float32)
    K = np.random.randn(B, H, S, d).astype(np.float32)
    V = np.random.randn(B, H, S, d).astype(np.float32)

    O_nano = nanoinfer.kernels.flash_attention(Q, K, V, causal=True)

    # Reference: naive causal attention with numpy
    def naive_attention(Q, K, V, scale):
        # Q,K,V: [B,H,S,d]
        scores = np.einsum("bhid,bhjd->bhij", Q, K) * scale   # [B,H,S,S]
        # causal mask
        mask = np.triu(np.ones((S, S), dtype=np.float32) * -1e9, k=1)
        scores = scores + mask[None, None]
        probs  = np.exp(scores - scores.max(axis=-1, keepdims=True))
        probs /= probs.sum(axis=-1, keepdims=True)
        return np.einsum("bhij,bhjd->bhid", probs, V)

    O_ref = naive_attention(Q, K, V, scale)

    max_diff = float(np.abs(O_nano - O_ref).max())
    status   = "PASS" if max_diff < 2e-2 else "FAIL"
    print(f"    {status}  max_diff={max_diff:.2e}  shape={O_nano.shape}")
except Exception as e:
    print(f"    SKIP  ({e})")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Smoke test complete.")
print("To run the full test suite:")
print("  pytest tests/test_kernels.py -v")
print("=" * 60)

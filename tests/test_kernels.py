"""
test_kernels.py — Python smoke tests for the nanoinfer.kernels sub-module.

Requires: nanoinfer .so to be built first.
    # Build via CMake:
    mkdir build && cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release -DCUDA_ARCHITECTURES=80
    make -j$(nproc)
    # OR via pip:
    pip install -e .

Run:
    pytest tests/test_kernels.py -v
    pytest tests/test_kernels.py -v --tb=short   # shorter tracebacks
"""

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Skip all tests gracefully if nanoinfer hasn't been built yet
# ---------------------------------------------------------------------------
nanoinfer = pytest.importorskip(
    "nanoinfer",
    reason="nanoinfer module not found — build first: pip install -e . or cmake+make",
)
kernels = nanoinfer.kernels


# ---------------------------------------------------------------------------
# Utility: relative tolerance comparison
# ---------------------------------------------------------------------------
def allclose(a: np.ndarray, b: np.ndarray, atol: float = 1e-3, rtol: float = 1e-3) -> bool:
    return bool(np.allclose(a, b, atol=atol, rtol=rtol))


# ===========================================================================
# Test group 1: kernels.gemm
# ===========================================================================

class TestGEMM:
    """nanoinfer.kernels.gemm(A, B) → C=A@B, both float32 numpy arrays."""

    def _run(self, M: int, K: int, N: int, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        A = rng.uniform(-1, 1, (M, K)).astype(np.float32)
        B = rng.uniform(-1, 1, (K, N)).astype(np.float32)

        C_ref = A @ B                       # numpy reference
        C_gpu = kernels.gemm(A, B)          # NanoInfer GPU kernel

        assert C_gpu.shape == (M, N), f"shape mismatch: {C_gpu.shape} != ({M},{N})"
        assert C_gpu.dtype == np.float32
        max_diff = float(np.max(np.abs(C_ref - C_gpu)))
        assert max_diff < 1e-3, (
            f"GEMM({M},{K},{N}): max_abs_diff={max_diff:.2e} > 1e-3"
        )

    # Small (TILE_16 path)
    def test_small_square(self):         self._run(16,  16,  16,  seed=1)
    def test_mid_square(self):           self._run(64,  64,  64,  seed=2)
    def test_non_square(self):           self._run(32,  64,  16,  seed=3)
    def test_non_square_tall(self):      self._run(128, 32,  64,  seed=4)
    def test_k_non_tile_aligned(self):   self._run(32,  37,  32,  seed=5)

    # Large (TILE_32 path)
    def test_large_square(self):         self._run(512, 512, 512, seed=6)
    def test_gpt2_attn(self):            self._run(512, 768, 768, seed=7)
    def test_gpt2_ffn_up(self):          self._run(512, 768, 3072, seed=8)
    def test_gpt2_ffn_down(self):        self._run(512, 3072, 768, seed=9)

    # Input validation
    def test_raises_on_1d(self):
        with pytest.raises(RuntimeError):
            kernels.gemm(np.ones(4, dtype=np.float32),
                         np.ones((4, 4), dtype=np.float32))

    def test_raises_on_dim_mismatch(self):
        with pytest.raises(RuntimeError):
            kernels.gemm(np.ones((3, 4), dtype=np.float32),
                         np.ones((5, 6), dtype=np.float32))  # K=4 vs 5


# ===========================================================================
# Test group 2: kernels.softmax
# ===========================================================================

class TestSoftmax:
    """nanoinfer.kernels.softmax(x) → row-wise online softmax, float32."""

    def _run(self, rows: int, cols: int, seed: int = 0,
             lo: float = -5.0, hi: float = 5.0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        x = rng.uniform(lo, hi, (rows, cols)).astype(np.float32)
        y = kernels.softmax(x)
        return x, y

    def _check(self, x: np.ndarray, y: np.ndarray, atol: float = 1e-4) -> None:
        assert y.shape == x.shape
        assert y.dtype == np.float32

        # All values in [0, 1] and finite
        assert np.all(np.isfinite(y)), "output contains NaN or Inf"
        assert np.all(y >= -1e-6),     "output has negative values"
        assert np.all(y <= 1 + 1e-6),  "output has values > 1"

        # Each row sums to 1
        row_sums = y.sum(axis=1)
        max_err  = float(np.max(np.abs(row_sums - 1.0)))
        assert max_err < atol, f"max row-sum error = {max_err:.2e} > {atol}"

    def _compare_numpy(self, x: np.ndarray, y: np.ndarray, atol: float = 1e-4) -> None:
        # NumPy stable softmax reference
        x_max = x.max(axis=1, keepdims=True)
        exp_x = np.exp(x - x_max)
        y_ref = exp_x / exp_x.sum(axis=1, keepdims=True)
        max_diff = float(np.max(np.abs(y - y_ref)))
        assert max_diff < atol, f"vs numpy: max_diff={max_diff:.2e}"

    # Correctness
    def test_single_element(self):
        x, y = self._run(1, 1, seed=0)
        self._check(x, y)
        assert abs(float(y[0, 0]) - 1.0) < 1e-6

    def test_single_row(self):
        x, y = self._run(1, 256, seed=1)
        self._check(x, y)
        self._compare_numpy(x, y)

    def test_attention_512(self):
        x, y = self._run(512, 512, seed=2)
        self._check(x, y)
        self._compare_numpy(x, y)

    def test_attention_1024(self):
        x, y = self._run(1024, 1024, seed=3)
        self._check(x, y)
        self._compare_numpy(x, y)

    def test_vocab_single_token(self):
        x, y = self._run(1, 50257, seed=4)
        self._check(x, y)

    def test_vocab_batch8(self):
        x, y = self._run(8, 50257, seed=5)
        self._check(x, y)

    # Numerical stability with large values (safe/online must not overflow)
    def test_large_values(self):
        rng = np.random.default_rng(99)
        x = rng.uniform(-90, 90, (8, 128)).astype(np.float32)
        y = kernels.softmax(x)
        self._check(x, y)

    def test_extreme_values(self):
        rng = np.random.default_rng(100)
        x = rng.uniform(-500, 500, (4, 64)).astype(np.float32)
        y = kernels.softmax(x)
        self._check(x, y)

    # Input validation
    def test_raises_on_1d(self):
        with pytest.raises(RuntimeError):
            kernels.softmax(np.ones(16, dtype=np.float32))

    def test_raises_on_3d(self):
        with pytest.raises(RuntimeError):
            kernels.softmax(np.ones((2, 4, 8), dtype=np.float32))


# ===========================================================================
# Test group 3: kernels.flash_attention
# ===========================================================================

class TestFlashAttention:
    """
    nanoinfer.kernels.flash_attention(Q, K, V, causal) → O
    Shapes: (B, H, S, d_head), all float32.
    """

    RNG = np.random.default_rng(42)

    @staticmethod
    def _make_qkv(B: int, H: int, S: int, d: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        # Small values to prevent softmax overflow in CPU reference
        Q = rng.uniform(-0.1, 0.1, (B, H, S, d)).astype(np.float32)
        K = rng.uniform(-0.1, 0.1, (B, H, S, d)).astype(np.float32)
        V = rng.uniform(-0.1, 0.1, (B, H, S, d)).astype(np.float32)
        return Q, K, V

    @staticmethod
    def _cpu_attention(Q, K, V, causal: bool) -> np.ndarray:
        """Numpy naive attention for reference."""
        B, H, S, d = Q.shape
        scale = 1.0 / np.sqrt(d)
        O = np.zeros_like(Q)
        for b in range(B):
            for h in range(H):
                q = Q[b, h]       # [S, d]
                k = K[b, h]
                v = V[b, h]
                scores = (q @ k.T) * scale   # [S, S]
                if causal:
                    mask = np.triu(np.ones((S, S), dtype=bool), k=1)
                    scores[mask] = -1e9
                # Stable softmax
                scores -= scores.max(axis=1, keepdims=True)
                probs = np.exp(scores)
                probs /= probs.sum(axis=1, keepdims=True)
                O[b, h] = probs @ v
        return O

    def _test(self, B: int, H: int, S: int, d: int,
               causal: bool, tol: float, seed: int = 0) -> None:
        Q, K, V = self._make_qkv(B, H, S, d, seed)

        O_ref = self._cpu_attention(Q, K, V, causal)
        O_gpu = kernels.flash_attention(Q, K, V, causal)

        assert O_gpu.shape == (B, H, S, d), f"shape mismatch: {O_gpu.shape}"
        assert O_gpu.dtype == np.float32
        assert np.all(np.isfinite(O_gpu)), "output contains NaN or Inf"

        max_diff = float(np.max(np.abs(O_ref - O_gpu)))
        assert max_diff < tol, (
            f"flash_attention B={B},H={H},S={S},d={d},causal={causal}: "
            f"max_diff={max_diff:.3e} > tol={tol:.0e}"
        )

    # Causal
    def test_causal_single_head_small(self):  self._test(1, 1,  16, 64, True,  2e-2, seed=1)
    def test_causal_single_head_32(self):     self._test(1, 1,  32, 64, True,  2e-2, seed=2)
    def test_causal_single_head_64(self):     self._test(1, 1,  64, 64, True,  2e-2, seed=3)
    def test_causal_gpt2_s64(self):           self._test(1, 12, 64, 64, True,  2e-2, seed=4)
    def test_causal_gpt2_s128(self):          self._test(1, 12,128, 64, True,  2e-2, seed=5)

    # Non-causal
    def test_noncausal_small(self):           self._test(1, 1,  32, 64, False, 2e-2, seed=6)
    def test_noncausal_multihead(self):       self._test(1, 4,  64, 64, False, 2e-2, seed=7)

    # Multi-batch
    def test_batch2(self):                    self._test(2, 1,  32, 64, True,  2e-2, seed=8)
    def test_batch4_gpt2(self):               self._test(4, 12, 64, 64, True,  2e-2, seed=9)

    # Edge: single token
    def test_single_token(self):              self._test(1, 1,   1, 64, True,  1e-5, seed=10)
    def test_single_token_multihead(self):    self._test(1, 12,  1, 64, True,  1e-5, seed=11)

    # Input validation
    def test_raises_on_3d(self):
        Q = np.ones((1, 4, 64), dtype=np.float32)  # should be 4-D
        with pytest.raises(RuntimeError):
            kernels.flash_attention(Q, Q, Q)

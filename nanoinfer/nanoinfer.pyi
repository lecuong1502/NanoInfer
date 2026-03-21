"""
nanoinfer.pyi — type stubs for the nanoinfer C++ extension.

These stubs give IDEs (VS Code, PyCharm) full autocompletion and type checking
for the pybind11 module without needing to parse C++ headers.

Install via:
    # stubs are picked up automatically when placed next to the .so
    # or inside the package directory alongside __init__.py
"""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# nanoinfer.kernels sub-module
# ---------------------------------------------------------------------------

class kernels:
    @staticmethod
    def gemm(
        A: NDArray[np.float32],
        B: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """
        Tiled GEMM: C = A @ B using a custom shared-memory CUDA kernel.

        Args:
            A: Shape (M, K), dtype float32.
            B: Shape (K, N), dtype float32.

        Returns:
            C: Shape (M, N), dtype float32.

        Raises:
            RuntimeError: If inputs are not 2-D or inner dimensions mismatch.
        """
        ...

    @staticmethod
    def softmax(
        x: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """
        Row-wise online softmax (single-pass, numerically stable).

        Args:
            x: Shape (rows, cols), dtype float32.

        Returns:
            y: Same shape as x; each row sums to 1.0.
        """
        ...

    @staticmethod
    def flash_attention(
        Q: NDArray[np.float32],
        K: NDArray[np.float32],
        V: NDArray[np.float32],
        causal: bool = True,
    ) -> NDArray[np.float32]:
        """
        Flash Attention v1 — IO-aware exact attention.

        Args:
            Q: Shape (batch, heads, seq_len, d_head), dtype float32.
            K: Shape (batch, heads, seq_len, d_head), dtype float32.
            V: Shape (batch, heads, seq_len, d_head), dtype float32.
            causal: Apply causal mask for autoregressive generation (default True).

        Returns:
            O: Same shape as Q.
        """
        ...


# ---------------------------------------------------------------------------
# nanoinfer.NanoInfer
# ---------------------------------------------------------------------------

class NanoInfer:
    """
    Lightweight CUDA inference engine for GPT-2 style transformer models.

    Example::

        model = NanoInfer.from_pretrained("gpt2", precision="int8")
        text  = model.generate("The quick brown fox", max_tokens=50)
        print(text)
    """

    def __init__(
        self,
        model_path: str,
        precision: str = "fp32",
        max_seq_len: int = 2048,
        kvcache_pages: int = 256,
    ) -> None:
        """
        Load model weights from a local directory.

        Args:
            model_path:    Path to directory containing model weights.
            precision:     One of ``"fp32"``, ``"fp16"``, ``"int8"``.
            max_seq_len:   Maximum sequence length supported (affects KV-cache size).
            kvcache_pages: Number of pages to allocate in the paged KV-cache.
        """
        ...

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        precision: str = "fp32",
        max_seq_len: int = 2048,
        kvcache_pages: int = 256,
    ) -> "NanoInfer":
        """
        Load a model by name or local path.

        Args:
            model_name:    Model name (e.g. ``"gpt2"``) or local directory path.
            precision:     One of ``"fp32"``, ``"fp16"``, ``"int8"``.
            max_seq_len:   Maximum sequence length.
            kvcache_pages: KV-cache page budget.

        Returns:
            A ready-to-use NanoInfer instance.
        """
        ...

    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 50,
        seed: int = -1,
    ) -> str:
        """
        Generate text continuation from a prompt.

        Args:
            prompt:      Input text string.
            max_tokens:  Maximum number of new tokens to generate.
            temperature: Sampling temperature. 0 = greedy, 1 = default, >1 = more random.
            top_p:       Nucleus sampling probability threshold (0–1).
            top_k:       Top-k vocabulary cutoff. 0 disables top-k filtering.
            seed:        Integer RNG seed for reproducibility. -1 = random seed.

        Returns:
            Generated continuation (does not include the original prompt).
        """
        ...

    def encode(self, text: str) -> list[int]:
        """Tokenize *text* and return a list of integer token ids."""
        ...

    def decode(self, ids: list[int]) -> str:
        """Convert a list of token ids back to a string."""
        ...

    # ---- read-only properties ----

    @property
    def precision(self) -> str:
        """Active precision: ``"fp32"``, ``"fp16"``, or ``"int8"``."""
        ...

    @property
    def vocab_size(self) -> int:
        """Vocabulary size of the loaded model."""
        ...

    @property
    def num_layers(self) -> int:
        """Number of transformer layers."""
        ...

    @property
    def num_heads(self) -> int:
        """Number of attention heads per layer."""
        ...

    @property
    def d_model(self) -> int:
        """Hidden dimension size."""
        ...

    def __repr__(self) -> str: ...

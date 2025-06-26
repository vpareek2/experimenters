# experimenters/embed/rope.py
import math
import torch
from typing import Optional


def _build_base_frequencies(head_dim: int, base: float) -> torch.Tensor:
    """
    Returns shape (head_dim // 2)
    w_i = 1 / (base ^ (2i / head_dim))
    """
    return 1.0 / (
        base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )


def precompute_rope_freqs(
    max_seq_len: int,
    head_dim: int,
    base: float = 10_000.0,
    original_ctx: Optional[int] = None,
    yarn_factor: float = 1.0,
    beta_fast: int = 32,
    beta_slow: int = 1,
) -> torch.Tensor:
    """
    Pre-computes the complex rotation factors (cos and sin) for each position.
    If max_seq_len exceedes original_ctx, we optionally apply YaRN scaline
    (long-context trick).

    Returns a tensor of shape (max_seq_len, head_dim // 2) with dtype=complex64.
    """
    freqs = _build_base_frequencies(head_dim, base)  # (head_dim // 2)
    if original_ctx and max_seq_len > original_ctx:
        # YaRN correction
        def _corr_dim(num_rot: int) -> torch.Tensor:
            return (
                head_dim
                * math.log(original_ctx / (num_rot * 2 * math.pi))
                / (2 * math.log(base))
            )

        def _range(low_rot: float, high_rot: float) -> torch.Tensor:
            lo = math.floor(_corr_dim(low_rot))
            hi = math.ceil(_corr_dim(high_rot))
            return max(lo, 0), min(hi, head_dim // 2 - 1)

        low, high = _range(beta_fast, beta_slow)
        ramp = torch.linspace(0, 1, head_dim // 2)
        smooth = torch.clamp((ramp - low) / max(high - low, 1e-5), 0, 1)
        freqs = freqs / yarn_factor * (1 - smooth) + freqs * smooth

    t = torch.arange(max_seq_len, dtype=freqs.dtype)  # (T,)
    freqs_outer = torch.outer(t, freqs)  # (T, head_dim // 2)
    return torch.polar(torch.ones_like(freqs_outer), freqs_outer)  # complex64


def apply_rotary_emb(
    x: torch.Tensor,  # (B, H, T, D)
    freqs_cis: torch.Tensor,  # (T, head_dim // 2) complex64
) -> torch.Tensor:
    """
    Applies RoPE in-place on the last two dims (rotates every pair).
    """
    B, H, T, D = x.size()
    assert D % 2 == 0, "RoPE requires even hidden dim"
    assert freqs_cis.size(-1) == x.size(-1) // 2, "RoPE table dims must be head_dim/2"

    # Cast to complex, rotate, cast back
    x_ = x.float().view(B, H, T, D // 2, 2)  # Split into real/imag parts
    x_complex = torch.view_as_complex(x_)
    freqs_complex = freqs_cis[:T].view(1, 1, T, -1)  # Broadcast B, H axes
    x_rot = x_complex * freqs_complex
    x_out = torch.view_as_real(x_rot).flatten(-2)  # Back to real tuple
    return x_out.to(dtype=x.dtype)

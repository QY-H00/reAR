import torch
from typing import Optional, Tuple
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend
from einops import rearrange

# ------------------------------------------------------
# Shuffle and Unshuffle
# ------------------------------------------------------

def shuffle(x, orders):
    batch_size, _ = x.shape[:2]
    batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, orders.shape[1])
    shuffled_x = x[batch_indices, orders]
    return shuffled_x

def unshuffle(shuffled_x, orders):
    # Unshuffle the tensor based on the original orders
    batch_size, seq_len = shuffled_x.shape[:2]
    batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, seq_len)
    unshuffled_x = torch.zeros_like(shuffled_x)
    unshuffled_x[batch_indices, orders] = shuffled_x
    return unshuffled_x

# ------------------------------------------------------
# 1D RoPE
# ------------------------------------------------------

def precompute_freqs_cis(dim: int, max_position_embeddings: int, theta: float = 10000.0, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Create the precomputed cos/sin for rotary embeddings (dim must be even).
    Returns a [max_position_embeddings, dim/2, 2] tensor with cos/sin.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=dtype) / dim))
    t = torch.arange(max_position_embeddings, dtype=dtype)
    freqs = torch.einsum('i,j->ij', t, freqs)  # [max_position_embeddings, dim/2]
    sin, cos = freqs.sin(), freqs.cos()
    # Combine cos/sin into last dimension
    return torch.stack([cos, sin], dim=-1)  # [max_pos, dim/2, 2]

def apply_rotary_emb(
    q: Optional[torch.Tensor],
    k: torch.Tensor,
    freqs_cis: torch.Tensor,
    orders: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    q, k: [B, n_heads, S, head_dim], head_dim must be even, make sure q and k corresponds to the spatial tokens (no cls, no bos)
    freqs_cis: [max_seq_len, head_dim/2, 2]
    orders: [B, S]
    """
    bsz, n_heads, seq_len, head_dim = k.shape
    bsz, n_heads, seq_len_q, head_dim = q.shape
    q = rearrange(q, "b n s d -> b s n d")
    k = rearrange(k, "b n s d -> b s n d")
    # slice out the needed positions
    assert freqs_cis.shape[0] >= seq_len, f"freqs_cis.shape[0] = {freqs_cis.shape[0]} must be >= seq_len = {seq_len}"
    # freqs_cis = freqs_cis[:seq_len]  # shape [seq_len, head_dim//2, 2]
    # Expand to shape [1, seq_len, 1, head_dim//2, 2]
    freqs_cis_len = freqs_cis.shape[0]
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)
    freqs_cis = freqs_cis.expand(bsz, freqs_cis_len, n_heads, head_dim // 2, 2)
    freqs_cis = shuffle(freqs_cis, orders)
    freqs_complex = torch.view_as_complex(freqs_cis.to(torch.float32))
    if seq_len_q < seq_len:
        freqs_complex_q = freqs_complex[:, -seq_len_q:]
    else:
        freqs_complex_q = freqs_complex

    # reshape Q/K to complex
    q_reshaped = q.view(bsz, seq_len_q, n_heads, head_dim // 2, 2)
    q_complex = torch.view_as_complex(q_reshaped.to(torch.float32))
    q_out = torch.view_as_real(q_complex * freqs_complex_q).to(q.dtype)
    q_out = q_out.view(bsz, seq_len_q, n_heads, head_dim)
    
    k_reshaped = k.view(bsz, seq_len, n_heads, head_dim // 2, 2)
    k_complex = torch.view_as_complex(k_reshaped.to(torch.float32))
    k_out = torch.view_as_real(k_complex * freqs_complex).to(k.dtype)
    k_out = k_out.view(bsz, seq_len, n_heads, head_dim)
    
    q_out = rearrange(q_out, "b s n d -> b n s d")
    k_out = rearrange(k_out, "b s n d -> b n s d")

    return q_out, k_out


def precompute_freqs_cis_2d(dim: int, height: int, width: int, theta: float) -> torch.Tensor:
    """
    Copied from https://github.com/mistralai/mistral-inference/blob/main/src/mistral_inference/rope.py

    freqs_cis: 2D complex tensor of shape (height, width, dim // 2) to be indexed by
        (height, width) position tuples
    """
    # (dim / 2) frequency bases
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))

    h = torch.arange(height, device=freqs.device)
    w = torch.arange(width, device=freqs.device)

    freqs_h = torch.outer(h, freqs[::2]).float()
    freqs_w = torch.outer(w, freqs[1::2]).float()
    freqs_2d = torch.cat(
        [
            freqs_h[:, None, :].repeat(1, width, 1),
            freqs_w[None, :, :].repeat(height, 1, 1),
        ],
        dim=-1,
    )

    # (height, width, dim // 2)
    freqs_2d = torch.polar(torch.ones_like(freqs_2d), freqs_2d) # (height, width, dim // 2)
    cos, sin = freqs_2d.real, freqs_2d.imag   
    # Combine cos/sin into last dimension
    return torch.stack([cos, sin], dim=-1)  # [height, width, dim //2, 2]


def apply_rotary_emb_2d(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs_cis: torch.Tensor,
    orders: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Copied from https://github.com/mistralai/mistral-inference/blob/main/src/mistral_inference/rope.py

    q, k: [B, n_heads, hw, head_dim], head_dim must be even, assume square
    freqs_cis: [max_seq_len, head_dim/2, 2]
    orders: [B, h * w]
    """
    bsz, n_heads, hw_q, head_dim = q.shape
    bsz, n_heads, hw, head_dim = k.shape
    h, w = int(hw ** 0.5), int(hw ** 0.5)
    q = rearrange(q, "b n s d -> b s n d")
    k = rearrange(k, "b n s d -> b s n d")
    # slice out the needed positions
    assert freqs_cis.shape[0] >= h, f"freqs_cis.shape[0] = {freqs_cis.shape[0]} must be >= h = {h}"
    assert freqs_cis.shape[1] >= w, f"freqs_cis.shape[1] = {freqs_cis.shape[1]} must be >= w = {w}"
    freqs_cis_h = freqs_cis.shape[0]
    freqs_cis_w = freqs_cis.shape[1]

    # Expand to shape [1, h, w, 1, head_dim//2, 2]
    freqs_cis = freqs_cis.unsqueeze(2).unsqueeze(0)

    # # reshape Q/K to complex
    q_reshaped = q.view(bsz, hw_q, n_heads, head_dim // 2, 2)
    k_reshaped = k.view(bsz, hw, n_heads, head_dim // 2, 2)

    # This convert is to ensure view_as_complex is supported
    q_complex = torch.view_as_complex(q_reshaped.to(torch.float32))
    k_complex = torch.view_as_complex(k_reshaped.to(torch.float32))

    # Properly expand freqs_cis to match the batch and head dimensions
    # [1, h, w, 1, head_dim//2, 2] -> [bsz, h, w, n_heads, head_dim//2, 2]
    freqs_cis = freqs_cis.expand(bsz, freqs_cis_h, freqs_cis_w, n_heads, head_dim // 2, 2)
    freqs_cis = rearrange(freqs_cis, "b h w n d t -> b (h w) n d t")
    freqs_cis = shuffle(freqs_cis, orders)
    freqs_complex = torch.view_as_complex(freqs_cis.to(torch.float32))
    if hw_q <= hw:
        freqs_complex_q = freqs_complex[:, -hw_q:]
    else:
        freqs_complex_q = freqs_complex

    q_out = torch.view_as_real(q_complex * freqs_complex_q).to(q.dtype)
    k_out = torch.view_as_real(k_complex * freqs_complex).to(k.dtype)

    q_out = q_out.view(bsz, hw_q, n_heads, head_dim)
    k_out = k_out.view(bsz, hw, n_heads, head_dim)
    q_out = rearrange(q_out, "b s n d -> b n s d")
    k_out = rearrange(k_out, "b s n d -> b n s d")
    return q_out, k_out

# ------------------------------------------------------
# Attention with RoPE
# ------------------------------------------------------

class ShuffledRoPEAttention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
            max_seq_len: int = 258,
            rope_type: str = '1d',
            theta: float = 10000.0,
            prefix: int = 2
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = True
        self.rope_type = rope_type

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.kv_cache = False
        self.k_cache = None
        self.v_cache = None

        if self.rope_type == '1d':
            self.register_buffer(
                'freqs_cis',
                precompute_freqs_cis(self.head_dim, max_seq_len, theta),
                persistent=False
            )
        elif self.rope_type == '2d':
            max_seq_len_sqr = int(max_seq_len ** 0.5)
            self.register_buffer(
                'freqs_cis_2d',
                precompute_freqs_cis_2d(self.head_dim, max_seq_len_sqr, max_seq_len_sqr, theta),
                persistent=False
            )
        else:
            raise ValueError(f"Invalid rope_type: {self.rope_type}")

        self.prefix = prefix

    def reset_kv_cache(self):
        self.k_cache = None
        self.v_cache = None

    @torch.compile
    def forward(self, x: torch.Tensor, orders: torch.Tensor, attn_mask=None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.kv_cache:
            if self.k_cache is None and self.v_cache is None:
                k_cache = k
                v_cache = v
            else:
                # assert N in [1, 2], f"x.shape {x.shape}"
                k_cache = torch.cat([self.k_cache, k], dim=-2)
                v_cache = torch.cat([self.v_cache, v], dim=-2)

            self.k_cache = k_cache
            self.v_cache = v_cache

            k = k_cache
            v = v_cache

        if self.rope_type not in ['1d', '2d']:
            raise ValueError(f"Invalid rope_type: {self.rope_type}")
        elif self.rope_type == '1d' and k.shape[2] > self.prefix:
            if self.kv_cache: 
                q_1d = q
            else: # during training
                q_1d = q[:, :, self.prefix:]
            k_1d = k[:, :, self.prefix:]
            q_1d, k_1d = apply_rotary_emb(q_1d, k_1d, self.freqs_cis, orders)
            if self.kv_cache: # during training
                q = q_1d
            else:
                q = torch.cat([q[:, :, :self.prefix], q_1d], dim=2)
            k = torch.cat([k[:, :, :self.prefix], k_1d], dim=2)
        elif self.rope_type == '2d' and k.shape[2] > self.prefix:
            if self.kv_cache:
                q_2d = q
            else: # during training
                q_2d = q[:, :, self.prefix:]
            k_2d = k[:, :, self.prefix:]
            q_2d, k_2d = apply_rotary_emb_2d(q_2d, k_2d, self.freqs_cis_2d, orders)
            if self.kv_cache:
                q = q_2d
            else:
                q = torch.cat([q[:, :, :self.prefix], q_2d], dim=2)
            k = torch.cat([k[:, :, :self.prefix], k_2d], dim=2)
        else:
            pass
        
        x = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            dropout_p=self.attn_drop.p if self.training else 0.,
        )
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

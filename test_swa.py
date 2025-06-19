from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# --- Dummy Config for standalone execution ---
class Config:
    def __init__(self):
        self.hidden_size = 512
        self.num_attention_heads = 16
        self.max_sequence_length_1d = 1600 # Max length for 1D tokens or temporal RoPE
        self.max_sequence_length = 40 # Max spatial dimension for 2D (e.g., if 8x8, then 8)
        self.theta = 10000.0
        self.initializer_range = 0.02
        self.window_size = 4
        self.rms_norm_eps = 1e-5
        self.intermediate_size = 1024 # For MLP

# ------------------------------------------------------
# RMSNorm
# ------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._norm(x) * self.weight

# ------------------------------------------------------
# 1D Rotary Positional Embedding (RoPE)
# ------------------------------------------------------

def precompute_freqs_cis(dim: int, max_position_embeddings: int, theta: float = 10000.0, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Precomputes 1D complex frequencies for RoPE."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=dtype) / dim))
    t = torch.arange(max_position_embeddings, dtype=dtype)
    freqs = torch.einsum('i,j->ij', t, freqs)
    cos, sin = freqs.cos(), freqs.sin()
    return torch.stack([cos, sin], dim=-1) # [max_pos, dim/2, 2]


def apply_rotary_emb(q: Optional[torch.Tensor], k: torch.Tensor, freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Applies 1D RoPE to query and key tensors."""
    bsz, seq_len, n_heads, head_dim = k.shape
    
    # Slice freqs_cis to current sequence length
    if freqs_cis.shape[0] < seq_len:
        raise ValueError(f"freqs_cis.shape[0] ({freqs_cis.shape[0]}) must be >= seq_len ({seq_len})")
    freqs_cis = freqs_cis[:seq_len] # [seq_len, head_dim//2, 2]
    
    # Expand for broadcasting
    freqs_complex = torch.view_as_complex(freqs_cis.unsqueeze(0).unsqueeze(2).to(torch.float32))

    q_out = q  # If q is None, return None; otherwise will be overwritten
    if q is not None:
        q_reshaped = q.view(bsz, seq_len, n_heads, head_dim // 2, 2)
        q_complex = torch.view_as_complex(q_reshaped.to(torch.float32))
        q_out = torch.view_as_real(q_complex * freqs_complex).to(q.dtype).view(bsz, seq_len, n_heads, head_dim)
    
    k_reshaped = k.view(bsz, seq_len, n_heads, head_dim // 2, 2)
    k_complex = torch.view_as_complex(k_reshaped.to(torch.float32))
    k_out = torch.view_as_real(k_complex * freqs_complex).to(k.dtype).view(bsz, seq_len, n_heads, head_dim)

    return q_out, k_out

# ------------------------------------------------------
# 2D Rotary Positional Embedding (RoPE)
# ------------------------------------------------------

def precompute_freqs_cis_2d(dim: int, height: int, width: int, theta: float) -> torch.Tensor:
    """Precomputes 2D complex frequencies for 2D RoPE."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    h_coords, w_coords = torch.arange(height, device=freqs.device), torch.arange(width, device=freqs.device)
    freqs_h = torch.outer(h_coords, freqs[::2]).float()
    freqs_w = torch.outer(w_coords, freqs[1::2]).float()

    freqs_2d = torch.cat(
        [freqs_h[:, None, :].repeat(1, width, 1), freqs_w[None, :, :].repeat(height, 1, 1)], dim=-1
    )
    freqs_2d = torch.polar(torch.ones_like(freqs_2d), freqs_2d) # (height, width, dim // 2)
    return torch.stack([freqs_2d.real, freqs_2d.imag], dim=-1) # [height, width, dim //2, 2]


def apply_rotary_emb_2d(q: torch.Tensor, k: torch.Tensor, freqs_cis: torch.Tensor,) -> Tuple[torch.Tensor, torch.Tensor]:
    """Applies 2D RoPE to query and key tensors."""
    bsz, h, w, n_heads, head_dim = q.shape
    
    # Slice freqs_cis to current dimensions
    if freqs_cis.shape[0] < h or freqs_cis.shape[1] < w:
        raise ValueError(f"freqs_cis dimensions ({freqs_cis.shape[0]}, {freqs_cis.shape[1]}) "
                         f"must be >= H ({h}) and W ({w}) respectively.")
    freqs_cis = freqs_cis[:h, :w] # [h, w, head_dim//2, 2]

    # Expand for broadcasting and convert to complex
    freqs_complex = torch.view_as_complex(freqs_cis.unsqueeze(0).unsqueeze(3).to(torch.float32))

    # Reshape Q/K to complex, apply rotation, and reshape back
    q_reshaped = q.view(bsz, h, w, n_heads, head_dim // 2, 2)
    k_reshaped = k.view(bsz, h, w, n_heads, head_dim // 2, 2)

    q_complex = torch.view_as_complex(q_reshaped.to(torch.float32))
    k_complex = torch.view_as_complex(k_reshaped.to(torch.float32))

    q_out = torch.view_as_real(q_complex * freqs_complex).to(q.dtype).view(bsz, h, w, n_heads, head_dim)
    k_out = torch.view_as_real(k_complex * freqs_complex).to(k.dtype).view(bsz, h, w, n_heads, head_dim)
    
    return q_out, k_out


# ------------------------------------------------------
# Sliding Window Rotary Multihead Attention
# ------------------------------------------------------

class SlidingWindowRotaryMultiheadAttention(nn.Module):
    """
    Multihead attention with sliding window and 1D/2D RoPE support.
    """
    def __init__(self, config: Config, type: str = None):
        super().__init__()
        self.config = config
        self.head_dim = config.hidden_size // config.num_attention_heads

        assert type in ['encoder', 'decoder'], f"type must be 'encoder' or 'decoder', got {type}"
        self.type = type

        self.mha = nn.MultiheadAttention(config.hidden_size, config.num_attention_heads, batch_first=True)

        # Add temporal embeddings for sliding window keys
        self.temporal_embed = nn.Parameter(torch.empty(config.window_size, config.hidden_size))
        nn.init.normal_(self.temporal_embed, mean=0.0, std=config.initializer_range)

        self.register_buffer('freqs_cis',
                             precompute_freqs_cis(self.head_dim, config.max_sequence_length_1d, config.theta),
                             persistent=False)
        self.register_buffer('freqs_cis_2d',
                             precompute_freqs_cis_2d(self.head_dim, config.max_sequence_length, config.max_sequence_length, config.theta),
                             persistent=False)

        nn.init.normal_(self.mha.in_proj_weight, mean=0.0, std=config.initializer_range)
        nn.init.zeros_(self.mha.in_proj_bias)
        nn.init.normal_(self.mha.out_proj.weight, mean=0.0, std=config.initializer_range)

    def sliding_key_value(self, k: torch.Tensor, v: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Creates sliding windows for K and V along the temporal dimension."""
        B, T, N, E = k.shape
        if window_size < 1:
            raise ValueError(f"window_size must be >=1, got {window_size}")

        k_pad = F.pad(k, (0, 0, 0, 0, window_size - 1, 0)) # Pad temporal dim
        v_pad = F.pad(v, (0, 0, 0, 0, window_size - 1, 0))

        # Unfold to create windows, then reshape
        k_windows = k_pad.unfold(dimension=1, size=window_size, step=1).permute(0, 1, 4, 2, 3)
        v_windows = v_pad.unfold(dimension=1, size=window_size, step=1).permute(0, 1, 4, 2, 3)
        return k_windows.contiguous(), v_windows.contiguous()
    
    def forward(
        self, 
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None, # Confirmed to be None
        position_ids: Optional[Tuple[int, int, int]] = None
    ) -> torch.Tensor:
        B, S, E = hidden_states.shape

        qkv = F.linear(hidden_states, self.mha.in_proj_weight, self.mha.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)

        T, H, W = position_ids # Temporal, Height, Width dimensions
        assert S == 2 * H * W * T, f"S={S} != 2*H*W*T={2*H*W*T}"

        n_heads = self.mha.num_heads
        head_dim = self.head_dim

        q = q.reshape(B, S, n_heads, head_dim)
        k = k.reshape(B, S, n_heads, head_dim)
        # v will be reshaped later for sliding_key_value

        # Apply 1D RoPE on 1D tokens
        q_1d = q[:, :-T*H*W, :, :].reshape(B * T, -1, n_heads, head_dim)
        k_1d = k[:, :-T*H*W, :, :].reshape(B * T, -1, n_heads, head_dim)
        q_1d, k_1d = apply_rotary_emb(q_1d, k_1d, self.freqs_cis)
        
        # Apply 2D RoPE on 2D tokens
        q_2d = q[:, -T*H*W:, :, :].reshape(B * T, H, W, n_heads, head_dim)
        k_2d = k[:, -T*H*W:, :, :].reshape(B * T, H, W, n_heads, head_dim)
        q_2d, k_2d = apply_rotary_emb_2d(q_2d, k_2d, self.freqs_cis_2d)
        q_2d = rearrange(q_2d, "(b t) h w n d -> (b t) (h w) n d", b=B, t=T)
        k_2d = rearrange(k_2d, "(b t) h w n d -> (b t) (h w) n d", b=B, t=T)
        
        # Merge 1D and 2D tokens for Query
        q = torch.cat([q_1d, q_2d], dim=1) # (B*T), (2*H*W), n_heads, head_dim
        q_mha = q.reshape(B * T, 2 * H * W, E) # (B*T, total_tokens_per_frame, E)

        # Prepare K, V for sliding window - concatenate 1D and 2D early
        k_combined = torch.cat([k_1d, k_2d], dim=1).reshape(B, T, 2*H*W, E)  # [B, T, 2*H*W, E]
        v_combined = v.reshape(B, T, 2*H*W, E)

        # Apply sliding window
        k_sliding, v_sliding = self.sliding_key_value(k_combined, v_combined, self.config.window_size)

        # Apply temporal embeddings to sliding window Keys
        temporal_emb = self.temporal_embed[None, None, :, None, :]  # [1, 1, window_size, 1, hidden_size]
        k_sliding = k_sliding + temporal_emb
        k_sliding = k_sliding.view(B, T, self.config.window_size * 2*H*W, E)
        v_sliding = v_sliding.view(B, T, self.config.window_size * 2*H*W, E)

        # Reshape for MHA
        k_mha = rearrange(k_sliding, "b t s e -> (b t) s e")  # (B*T, window_size * 2*H*W, E)
        v_mha = rearrange(v_sliding, "b t s e -> (b t) s e")  # (B*T, window_size * 2*H*W, E)

        print(f"q_mha.shape: {q_mha.shape}, k_mha.shape: {k_mha.shape}, v_mha.shape: {v_mha.shape}, T: {T}, H: {H}, W: {W}, 2*H*W: {2*H*W}, window_size: {self.config.window_size}, dtype: {q_mha.dtype}")
        # Multihead Attention call (attention_mask is None)
        # with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
        out, _ = self.mha(q_mha, k_mha, v_mha, attn_mask=attention_mask, need_weights=False)
        
        out = out.reshape(B, S, E) # Reshape output to original B, S, E
        return out


# ------------------------------------------------------------------
# MLP
# ------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.w1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.w3 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)

        nn.init.normal_(self.w1.weight, mean=0.0, std=config.initializer_range)
        nn.init.normal_(self.w2.weight, mean=0.0, std=config.initializer_range)
        nn.init.normal_(self.w3.weight, mean=0.0, std=config.initializer_range)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


# ------------------------------------------------------------------
# SWATransformerBlock (Sliding Window Attention Transformer Block)
# ------------------------------------------------------------------

class SWATransformerBlock(nn.Module):
    """
    A Transformer block for Sliding Window Attention, including RMSNorm and MLP.
    """
    def __init__(self, config: Config, type: str = None):
        super().__init__()
        self.attention_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.attention = SlidingWindowRotaryMultiheadAttention(
            config, 
            type=type
        )
        self.mlp = MLP(config)

    def forward(
        self, 
        hidden_states: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None, # Confirmed to be None
        position_ids: Optional[Tuple[int, int, int]] = None,
    ) -> torch.Tensor:
        # Attention with pre-norm
        attn_in = self.attention_norm(hidden_states)
        attn_out = self.attention(attn_in, attention_mask=attention_mask, position_ids=position_ids)
        hidden_states = hidden_states + attn_out

        # MLP with pre-norm
        ffn_in = self.ffn_norm(hidden_states)
        ffn_out = self.mlp(ffn_in)
        hidden_states = hidden_states + ffn_out
        return hidden_states
    

# --- Test Script ---
def run_test():
    print("Running SWATransformerBlock test...")
    
    # 1. Instantiate Config
    config = Config()
    
    # Define input dimensions
    B = 1  # Batch size
    T = 13  # Temporal dimension
    H = 40  # Height dimension
    W = 40  # Width dimension
    E = config.hidden_size # Embedding dimension
    
    # Total sequence length (S = 2 * H * W * T for 1D and 2D tokens)
    S = 2 * H * W * T
    
    # 2. Instantiate SWATransformerBlock
    model = SWATransformerBlock(config, type='encoder').to("cuda")
    
    # 3. Create dummy input tensors
    hidden_states = torch.randn(B, S, E, dtype=torch.float32).to("cuda")
    position_ids = (T, H, W) # Passed as a tuple
    
    print(f"Input hidden_states shape: {hidden_states.shape}")
    print(f"Position IDs (T, H, W): {position_ids}")
    
    # 4. Perform forward pass
    output = model(hidden_states, attention_mask=None, position_ids=position_ids)

    from time import sleep
    sleep(1000)  # Sleep for 1 second
    
    # 5. Assert output shape
    expected_output_shape = (B, S, E)
    assert output.shape == expected_output_shape, \
        f"Test failed: Expected output shape {expected_output_shape}, but got {output.shape}"
    
    print(f"Test passed! Output shape: {output.shape}")

if __name__ == '__main__':
    run_test()

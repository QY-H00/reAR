import torch

def shuffle(x, orders):
    batch_size, seq_len = x.shape[:2]
    batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, orders.shape[1])
    shuffled_x = x[batch_indices, orders]
    return shuffled_x

def unshuffle(shuffled_x, orders):
    batch_size, seq_len = shuffled_x.shape[:2]
    batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, seq_len)
    unshuffled_x = torch.zeros_like(shuffled_x)
    unshuffled_x[batch_indices, orders] = shuffled_x
    return unshuffled_x

def sliding_window_shift(x, k_tokens):
    B, seq_len = x.shape[:2]
    i_indices = torch.arange(seq_len, device=x.device).unsqueeze(1)  # [seq_len, 1]
    k_indices = torch.arange(k_tokens, device=x.device).unsqueeze(0)  # [1, k_tokens]
    source_indices = i_indices + k_indices  # [seq_len, k_tokens]
    source_indices = torch.clamp(source_indices, 0, seq_len - 1)
    x_shifted = x[:, source_indices] # [B, seq_len, k_tokens, ...]
    return x_shifted

# Vectorized version without loops
def build_sliding_mask(N, k):
    """
    Vectorized version of the sliding mask creation for the dAR loss.
    """
    mask = torch.zeros(N, k)
    
    # Create row indices
    row_indices = torch.arange(N).unsqueeze(1)  # [N, 1]
    col_indices = torch.arange(k).unsqueeze(0)  # [1, k]
    
    # First N-k rows: all positions are i/N
    first_block_mask = row_indices < (N - k)  # [N, 1]
    mask = torch.where(first_block_mask, 1.0 / k, mask)
    
    # Last k rows: causal mask
    # For row i in [N-k, N-1], position j should be non-zero if j <= i - (N-k)
    causal_block_mask = row_indices >= (N - k)  # [N, 1]
    causal_position_mask = col_indices <= (row_indices - (N - k))  # [N, k]
    causal_mask = causal_block_mask & causal_position_mask  # [N, k]
    
    mask = torch.where(causal_mask, 1.0 / (N - row_indices), mask)
    
    return mask
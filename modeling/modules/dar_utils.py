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

# def sliding_window_shift(x: torch.Tensor, k_tokens: int) -> torch.Tensor:
#     """
#     Creates sliding windows of width k_tokens for each position in the sequence.
#     Edges are handled by replicating the last element so all windows are size k_tokens.

#     Args:
#         x: Tensor of shape [B, L, D1, D2, ...].
#         k_tokens: window size.

#     Returns:
#         Tensor of shape [B, L, k_tokens, D1, D2, ...].
#     """
#     permute_dims = list(range(x.dim()))
#     permute_dims = permute_dims[:1] + permute_dims[2:] + [1]
#     x_permuted = x.permute(permute_dims)
#     pad_size = k_tokens - 1
#     x_padded = torch.nn.functional.pad(x_permuted, (0, pad_size), mode='replicate')
#     windows = x_padded.unfold(dimension=-1, size=k_tokens, step=1)
#     out_permute = [0, -2, -1] + list(range(1, x.dim()-1))
#     out = windows.permute(out_permute)
#     return out

def sliding_window_shift(x, k_tokens, add_offset=0):
    B, seq_len = x.shape[:2]
    assert k_tokens <= seq_len, "k_tokens should be less than or equal to seq_len"
    i_indices = torch.arange(seq_len + add_offset, device=x.device).unsqueeze(1)  # [seq_len, 1]
    k_indices = torch.arange(k_tokens, device=x.device).unsqueeze(0)  # [1, k_tokens]
    source_indices = i_indices + k_indices  # [seq_len, k_tokens]
    source_indices = torch.clamp(source_indices, 0, seq_len - 1)
    x_shifted = x[:, source_indices] # [B, seq_len, k_tokens, ...]
    return x_shifted

def sum_reciprocal_series(m: torch.Tensor, n: int = 256, k: int = 4) -> torch.Tensor:
    """
    Compute S(m) = sum_{i=0..floor((n-m)/k)} 1/(m + i*k) for each m in the input tensor.
    
    Args:
        m: Tensor of shape (N,) or (N, 1), containing the starting terms.
        n: Maximum value to cap the series (default 256).
        k: Step size between terms (default 4).
    
    Returns:
        Tensor of shape (N,), where each entry is the sum S(m).
    """
    # Ensure m has shape (N, 1)
    if m.dim() == 1:
        m = m.unsqueeze(1)
    
    # Determine the maximal number of terms across all m
    m_min = int(m.min().item())
    t_max = (n - m_min) // k
    
    # Build index tensor for i = 0 .. t_max
    i = torch.arange(t_max + 1, device=m.device).unsqueeze(0)  # shape (1, t_max+1)
    
    # Compute all denominators: m + k*i
    denom = m + k * i  # shape (N, t_max+1)
    
    # Mask out terms where denom > n
    mask = denom <= n
    
    # Convert to float and compute reciprocals, zero out invalid terms
    denom = denom.float()
    terms = torch.where(mask, denom.reciprocal(), torch.zeros_like(denom))
    
    # Sum over the series dimension and return
    result = terms.sum(dim=1, keepdim=True)

    return result

# Vectorized version without loops
def build_sliding_mask(N, k):
    """
    Vectorized version of the sliding mask creation for the dAR loss.
    """
    mask = torch.zeros(N, k)
    
    # Create row indices
    row_indices = torch.arange(N).unsqueeze(1)  # [N, 1]
    col_indices = torch.arange(k).unsqueeze(0)  # [1, k]

    # scaling_factors = 1 / k
    
    # First N-k rows: all positions are i/N
    first_block_mask = row_indices < (N - k)  # [N, 1]
    mask = torch.where(first_block_mask, 1/k, mask)
    
    # Last k rows: causal mask
    # For row i in [N-k, N-1], position j should be non-zero if j <= i - (N-k)
    causal_block_mask = row_indices >= (N - k)  # [N, 1]
    causal_position_mask = col_indices <= (N - row_indices - 1)  # [N, k]
    causal_mask = causal_block_mask & causal_position_mask  # [N, k]
    
    mask = torch.where(causal_mask, 1 / (N - row_indices), mask)
    
    return mask

def update_full_order_vectorized_attempt(current_order, full_order):
    """
    Attempt at a more vectorized version, but still needs a loop for random shuffling per batch.
    """
    B, S1 = current_order.shape
    B_full, S = full_order.shape
    device = current_order.device
    
    # Copy current order
    full_order[:, :S1] = current_order
    
    # Create all possible indices for each batch
    all_indices = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)  # [B, S]
    
    # Create mask for used indices
    used_mask = torch.zeros(B, S, dtype=torch.bool, device=device)
    batch_indices = torch.arange(B, device=device).unsqueeze(1)
    used_mask[batch_indices, current_order] = True
    
    # Get unused indices (this creates ragged tensor, so we still need loop)
    for b in range(B):
        unused = all_indices[b][~used_mask[b]]
        perm_unused = unused[torch.randperm(len(unused), device=device)]
        full_order[b, S1:] = perm_unused
    
    return full_order
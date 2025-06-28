import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F

from .dar_utils import sliding_window_shift


class DiCubeHead(nn.Module):
    """Diffusion Head"""
    def __init__(
        self,
        hidden_size,
        mlp_size,
        vocab_size,
        seq_len,
        depth,
        k_tokens=1,
        grad_checkpointing=False
    ):
        super(DiCubeHead, self).__init__()
        self.in_channels = hidden_size
        self.prediction_head = SimpleMLPAdaLN(
            in_channels=hidden_size,
            model_channels=mlp_size,
            out_channels=vocab_size,
            z_channels=hidden_size, # Default as the same with hidden_size
            num_res_blocks=depth,
            seq_len=seq_len,
            grad_checkpointing=grad_checkpointing
        )
        self.seq_len = seq_len
        self.k_tokens = k_tokens

    def forward(self, x, orders, c, k_tokens=None):
        if k_tokens is None:
            k_tokens = self.k_tokens
        B, seq_len, hidden_size = x.shape # Notice that x is currently in shape of seq_len + 1
        orders = sliding_window_shift(orders, k_tokens, add_offset=1) # [B, seq_len, k_tokens]
        x = x.unsqueeze(2).expand(B, seq_len, k_tokens, hidden_size)  # [B, seq_len, k_tokens, hidden_size]
        c = c.unsqueeze(2).expand(B, seq_len, k_tokens, hidden_size)  # [B, seq_len, k_tokens, hidden_size]
        result = self.prediction_head(x, orders, c)
        return result

    def forward_inference(self, x, orders, c, k_tokens=None, full_orders=None):
        # orders indicate current orders of x, full_orders indicate the full prediction orders (including those not generated yet)
        if k_tokens is None:
            k_tokens = self.k_tokens
        B, seq_len, hidden_size = x.shape
        full_orders = sliding_window_shift(full_orders, k_tokens, add_offset=0) # [B, seq_len, k_tokens]
        orders = full_orders[:, orders.shape[1] - x.shape[1]+1:orders.shape[1]+1] # corresponding to the predicted targets of current x
        x = x.unsqueeze(2).expand(B, seq_len, k_tokens, hidden_size)  # [B, seq_len, k_tokens, hidden_size]
        c = c.unsqueeze(2).expand(B, seq_len, k_tokens, hidden_size)  # [B, seq_len, k_tokens, hidden_size]
        result = self.prediction_head(x, orders, c)
        return result


def apply_freqs_cis(cond_cache: torch.Tensor,
                       pos_idx: torch.Tensor) -> torch.Tensor:
    """
    Gather 2D-frequency embeddings for arbitrary position indices.

    Args:
        cond_cache: FloatTensor of shape [N_pos, dim], as returned by
                    precompute_freqs_cis_2d(...)
        pos_idx:    LongTensor (or castable) of shape [..., S], where each
                    element is in [0, N_pos).

    Returns:
        FloatTensor of shape [..., S, dim], where
        output[..., i, :] = cond_cache[pos_idx[..., i], :].
    """
    # make sure indices are integer and on same device
    idx = pos_idx.to(cond_cache.device).long()
    # use F.embedding (works with any leading shape)
    return F.embedding(idx, cond_cache)

def precompute_freqs_cis_1d(
    seq_len: int,
    dim: int,
    base: int = 10000,
    cls_token_num: int = 0
) -> torch.Tensor:
    """
    Build a 1D rotary-frequency cache.

    Args:
      seq_len:        length of the sequence (number of positions)
      dim:            total embedding dimension (must be even)
      base:           frequency base (default 10000)
      cls_token_num:  number of leading “class” tokens (zero vectors)

    Returns:
      cond_cache: FloatTensor of shape [cls_token_num + seq_len, dim].
                  Rows 0..cls_token_num-1 are zeros;
                  row cls_token_num + p is the cos/sin embedding for position p.
    """
    # dim must be even to form cos/sin pairs
    assert dim % 2 == 0, "dim must be an even number"

    half_dim = dim // 2  # number of frequency pairs
    # inverse frequencies: [1/base^(0/half), 1/base^(1/half), ..., 1/base^((half_dim-1)/half)]
    inv_freqs = 1.0 / (base ** (torch.arange(half_dim, dtype=torch.float32) / half_dim))
    # positions [0, 1, ..., seq_len-1]
    t = torch.arange(seq_len, dtype=torch.float32, device=inv_freqs.device)

    # outer product → (seq_len, half_dim)
    freqs = torch.outer(t, inv_freqs)

    # build (cos, sin) → shape (seq_len, half_dim, 2)
    emb = torch.stack([torch.cos(freqs), torch.sin(freqs)], dim=-1)

    # flatten last two dims → (seq_len, dim)
    emb = emb.reshape(seq_len, dim)

    # prepend cls_token_num rows of zeros if needed
    if cls_token_num > 0:
        zeros = torch.zeros(cls_token_num, dim, device=emb.device)
        emb = torch.cat([zeros, emb], dim=0)

    return emb  # shape: [cls_token_num + seq_len, dim]

def precompute_freqs_cis_2d(
    grid_size: int, n_elem: int, base: int = 10000, cls_token_num=120
):
    # split the dimension into half, one for x and one for y
    half_dim = n_elem // 2
    freqs = 1.0 / (
        base ** (torch.arange(0, half_dim, 2)[: (half_dim // 2)].float() / half_dim)
    )
    t = torch.arange(grid_size, device=freqs.device)
    freqs = torch.outer(t, freqs)  # (grid_size, head_dim // 2)
    freqs_grid = torch.concat(
        [
            freqs[:, None, :].expand(-1, grid_size, -1),
            freqs[None, :, :].expand(grid_size, -1, -1),
        ],
        dim=-1,
    )  # (grid_size, grid_size, head_dim // 2)
    cache_grid = torch.stack(
        [torch.cos(freqs_grid), torch.sin(freqs_grid)], dim=-1
    )  # (grid_size, grid_size, head_dim // 2, 2)
    cache = cache_grid.flatten(0, 1)
    cond_cache = torch.cat(
        [torch.zeros(cls_token_num, n_elem // 2, 2), cache]
    )  # (cls_token_num+grid_size**2, head_dim // 2, 2)
    return cond_cache.reshape(cond_cache.shape[0], -1)



def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class PositionEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, seq_len=256, frequency_embedding_size=768, pos_type="2d"):
        super().__init__()
        assert pos_type in ["1d", "2d"]
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        if pos_type == "1d":
            self.register_buffer(
                "freqs_cis",
                precompute_freqs_cis_1d(seq_len, frequency_embedding_size),
                persistent=False
            )
        else:
            self.register_buffer(
                "freqs_cis",
                precompute_freqs_cis_2d(seq_len, frequency_embedding_size),
                persistent=False
            )

    def position_embedding(self, orders):
        """
        Create sinusoidal timestep embeddings.
        :param orders: a [..., S] Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        pos_freqs = apply_freqs_cis(self.freqs_cis, orders)
        return pos_freqs

    def forward(self, orders):
        pos_freqs = self.position_embedding(orders)
        pos_emb = self.mlp(pos_freqs)
        return pos_emb


class ResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    """

    def __init__(
        self,
        channels
    ):
        super().__init__()
        self.channels = channels

        self.in_ln = nn.LayerNorm(channels, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels, bias=True),
            nn.SiLU(),
            nn.Linear(channels, channels, bias=True),
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channels, 3 * channels, bias=True)
        )

    def forward(self, x, y):
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(y).chunk(3, dim=-1)
        h = modulate(self.in_ln(x), shift_mlp, scale_mlp)
        h = self.mlp(h)
        return x + gate_mlp * h


class FinalLayer(nn.Module):
    """
    The final layer adopted from DiT.
    """
    def __init__(self, model_channels, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(model_channels, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(model_channels, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(model_channels, 2 * model_channels, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class SimpleMLPAdaLN(nn.Module):
    """
    The MLP for Diffusion Loss.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param z_channels: channels in the condition.
    :param num_res_blocks: number of residual blocks per downsample.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        z_channels,
        num_res_blocks,
        seq_len,
        grad_checkpointing=False
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.grad_checkpointing = grad_checkpointing

        self.pos_embed = PositionEmbedder(model_channels, seq_len=seq_len)
        self.cond_embed = nn.Linear(z_channels, model_channels)

        self.input_proj = nn.Linear(in_channels, model_channels)

        res_blocks = []
        for i in range(num_res_blocks):
            res_blocks.append(ResBlock(
                model_channels,
            ))

        self.res_blocks = nn.ModuleList(res_blocks)
        self.final_layer = FinalLayer(model_channels, out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize timestep embedding MLP
        nn.init.normal_(self.pos_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.pos_embed.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers
        for block in self.res_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    @torch.compile
    def forward(self, x, orders, c):
        """
        Apply the model to an input batch.
        :param x: an [N x C] Tensor of inputs.
        :param orders: a 1-D batch of timesteps.
        :param c: conditioning from AR transformer.
        :return: an [N x C] Tensor of outputs.
        """
        x = self.input_proj(x)
        orders = self.pos_embed(orders)
        c = self.cond_embed(c)

        y = orders + c

        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.res_blocks:
                x = checkpoint(block, x, y)
        else:
            for block in self.res_blocks:
                x = block(x, y)

        return self.final_layer(x, y)

    def forward_with_cfg(self, x, t, c, cfg_scale):
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, c)
        eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)
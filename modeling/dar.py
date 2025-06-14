"""This file contains the model definition of TiTok.

Copyright (2024) Bytedance Ltd. and/or its affiliates

Licensed under the Apache License, Version 2.0 (the "License"); 
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at 

    http://www.apache.org/licenses/LICENSE-2.0 

Unless required by applicable law or agreed to in writing, software 
distributed under the License is distributed on an "AS IS" BASIS, 
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
See the License for the specific language governing permissions and 
limitations under the License.

Reference: 
    https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/transformer.py
    https://github.com/facebookresearch/DiT/blob/main/models.py
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.modules import BaseModel
from functools import partial
from timm.layers import Mlp
import random

# util function
def build_causal_mask(seq_length):
    mask = torch.empty(seq_length, seq_length)
    mask.fill_(float("-inf"))
    mask.triu_(1)  # zero out the lower diagonal
    return mask

# weight init
def init_weights(module):
    if (isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d) or
     isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d)):
        module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        if module.bias is not None:
            module.bias.data.zero_()
        if module.weight is not None:
            module.weight.data.fill_(1.0)

class DiffusionHead(nn.Module):
    def __init__(self, dim, norm_layer, seq_len, vocab_size, k_tokens=4, type="simple"):
        super().__init__()
        self.type = type
        self.vocab_size = vocab_size
        if type == "simple":
            assert k_tokens == vocab_size, "k_tokens should be equal to vocab_size if use simple head"
            self.head = nn.Linear(dim, seq_len * vocab_size, bias=True)
        elif type == "reduced":
            self.pos_condition = nn.init.trunc_normal_(nn.Parameter(torch.zeros(1, seq_len, dim)), 0., 0.02)
            self.prediction_head = nn.Linear(dim, vocab_size, bias=True)
            self.norm_final = norm_layer(dim, elementwise_affine=False)
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(dim, 2*dim)
            )
            self.k_tokens = k_tokens
        else:
            raise ValueError(f"Invalid type: {type}")
        
    def forward(self, x, orders):
        # x: [B, seq_len, dim]
        # orders: [B, seq_len]
        
        if self.type == "simple":
            return self.head(x)
        elif self.type == "reduced":
            B, seq_len, dim = x.shape
            # Shuffle pos_condition according to orders
            pos_condition_expanded = self.pos_condition.expand(B, -1, -1)
            batch_indices = torch.arange(B, device=x.device).unsqueeze(1).expand(-1, seq_len)
            shuffled_pos_condition = pos_condition_expanded[batch_indices, orders]  # [B, seq_len, dim]
            
            # Compute scale and shift from shuffled position condition
            scale_shift = self.adaLN_modulation(shuffled_pos_condition)  # [B, seq_len, 2*dim]
            scale, shift = scale_shift.chunk(2, dim=-1)  # Each: [B, seq_len, dim]
            
            # Duplicate x for k_tokens times
            x_duplicated = x.unsqueeze(2).expand(-1, -1, self.k_tokens, -1)  # [B, seq_len, k_tokens, dim]
            
            # Modulate each token copy with scale and shift
            scale_expanded = scale.unsqueeze(2).expand(-1, -1, self.k_tokens, -1)  # [B, seq_len, k_tokens, dim]
            shift_expanded = shift.unsqueeze(2).expand(-1, -1, self.k_tokens, -1)  # [B, seq_len, k_tokens, dim]
            
            # Apply modulation: x * (1 + scale) + shift
            x_modulated = modulate(self.norm_final(x_duplicated), shift_expanded, scale_expanded)  # [B, seq_len, k_tokens, dim]
            
            # Reshape for prediction head: [B, seq_len, k_tokens, dim] -> [B, seq_len * k_tokens, dim]
            x_reshaped = x_modulated.reshape(B, seq_len * self.k_tokens, dim)
            
            # Forward through prediction head
            logits = self.prediction_head(x_reshaped)  # [B, seq_len * k_tokens, vocab_size]
            
            # Reshape to final output: [B, seq_len, k_tokens * vocab_size]
            output = logits.reshape(B, seq_len, self.k_tokens * self.vocab_size)
            
            return output

# attention layer with KV cache supported
class Attention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = True

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.kv_cache = False
        self.k_cache = None
        self.v_cache = None

    def reset_kv_cache(self):
        self.k_cache = None
        self.v_cache = None

    def forward(self, x: torch.Tensor, attn_mask=None) -> torch.Tensor:
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
        x = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            dropout_p=self.attn_drop.p if self.training else 0.,
        )
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

def modulate(x, shift, scale):
    return x * (1 + scale) + shift

class AdaLN(nn.Module):
    def __init__(self, dim, norm_layer):
        super().__init__()
        self.norm_final = norm_layer(dim, elementwise_affine=False)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 2*dim)
        )
    
    def forward(self, x, c):
        scale, shift = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        return x

# basic transformer block
class Block(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = Mlp,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        
        self.attn = Attention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True)
        )


    def forward(self, x: torch.Tensor, attn_mask=None, c = None) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), attn_mask=attn_mask)
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class dAR(BaseModel):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        # parse the configs
        embed_dim = config.model.generator.hidden_size
        depth = config.model.generator.num_hidden_layers
        num_heads = config.model.generator.num_attention_heads
        intermediate_size = config.model.generator.intermediate_size
        mlp_ratio = intermediate_size / embed_dim

        image_seq_len = config.model.generator.image_seq_len
        target_codebook_size = config.model.vq_model.codebook_size
        condition_num_classes = config.model.generator.condition_num_classes
        norm_layer=partial(nn.LayerNorm, eps=1e-6)

        dropout_rate = config.model.generator.dropout
        attn_dropout_rate = config.model.generator.attn_drop
   
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                qk_norm=True,
                proj_drop=dropout_rate,
                attn_drop=attn_dropout_rate,
                norm_layer=norm_layer)
            for i in range(depth)])

        self.embeddings = nn.Embedding(
            target_codebook_size + 1 + condition_num_classes + 1, embed_dim)

        self.pos_embed = nn.init.trunc_normal_(
            nn.Parameter(torch.zeros(1, image_seq_len + 1024, embed_dim)), 0., 0.02)

        # self.target_aware_pos_embed = nn.init.trunc_normal_(
        #     nn.Parameter(torch.zeros(1, image_seq_len + 1024, embed_dim)), 0., 0.02)

        # number of steps == image_seq_len
        self.timesteps_embeddings = nn.init.trunc_normal_(
            nn.Parameter(torch.zeros(1, image_seq_len + 100, embed_dim)), 0., 0.02)
        self.adaln_before_head = AdaLN(embed_dim, norm_layer=norm_layer)
        self.condition_num_classes = condition_num_classes
        self.image_seq_len = image_seq_len
        self.target_codebook_size = target_codebook_size
        self.none_condition_id = self.condition_num_classes + self.target_codebook_size + 1
        
        self.apply(init_weights)

        self.lm_head = nn.Linear(embed_dim,
            image_seq_len * target_codebook_size, bias=True)
        
        nn.init.trunc_normal_(self.lm_head.weight, mean=0.0, std=0.02) # lm_head weight seems to be instable
        nn.init.zeros_(self.lm_head.bias)

        attn_mask = build_causal_mask(self.image_seq_len + 1024) # include condition
        self.register_buffer('attn_mask', attn_mask, persistent=False)

        self.use_checkpoint = config.model.generator.get("use_checkpoint", False)

        # init for adaln-zero.

        nn.init.constant_(self.adaln_before_head.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaln_before_head.adaLN_modulation[-1].bias, 0)
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        self.random_ratio = 1.0

    def enable_kv_cache(self):
        for block in self.blocks:
            block.attn.kv_cache = True
            block.attn.reset_kv_cache()

    def disable_kv_cache(self):
        for block in self.blocks:
            block.attn.kv_cache = False
            block.attn.reset_kv_cache()

    def sample_orders(self, x):
        batch_size = x.shape[0]
        shuffled_orders = []

        for _ in range(batch_size):
            if random.random() < self.random_ratio:
                # random order
                shuffled_orders.append(torch.randperm(self.image_seq_len, device=x.device))
            else:
                # raster order
                shuffled_orders.append(torch.arange(self.image_seq_len, device=x.device))
                
        shuffled_orders = torch.stack(shuffled_orders)
        return shuffled_orders.to(x.device)
    
    def set_random_ratio(self, new_ratio):
        # Always set to 1.0 for dAR (different from RAR)
        self.random_ratio = 1.0

    def get_raster_orders(self, x):
        batch_size = x.shape[0]
        shuffled_orders = torch.stack([torch.arange(self.image_seq_len, device=x.device) for _ in range(batch_size)])
        return shuffled_orders

    def shuffle(self, x, orders):
        batch_size, seq_len = x.shape[:2]
        batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, orders.shape[1])
        shuffled_x = x[batch_indices, orders]
        return shuffled_x

    def unshuffle(self, shuffled_x, orders):
        # Unshuffle the tensor based on the original orders
        batch_size, seq_len = shuffled_x.shape[:2]
        batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, seq_len)
        unshuffled_x = torch.zeros_like(shuffled_x)
        unshuffled_x[batch_indices, orders] = shuffled_x
        return unshuffled_x

    def preprocess_condition(self, condition, cond_drop_prob=0.0):
        # Set class condition to None condition
        drop_label_mask = torch.rand_like(condition, dtype=torch.float) < cond_drop_prob
        condition = condition + self.target_codebook_size + 1  # [0, 999] -> [codebook_size + 1, codebook_size + 999]
        condition[drop_label_mask] = self.none_condition_id
        return condition

    def get_none_condition(self,
                           condition
                           ):
        return torch.full_like(condition, self.none_condition_id)
    
    def forward(self, input_ids, condition, return_labels=False):
        orders = self.sample_orders(input_ids)
        return self.forward_fn(input_ids, condition, return_labels, orders)

    def forward_fn(self, input_ids, condition,
                   return_labels=False,
                   orders=None,
                   is_sampling=False):
        # Token space:
        #  [0, codebook_size - 1]                       : those are the learned quantized image tokens
        #  codebook_size                                : the mask token used to mask image tokens
        #  [codebook_size + 1, codebook_size + nclass]  : the imagenet class tokens
        #  codebook_size + 1 + nclass                   : the class drop label

        if orders is None:
            orders = self.get_raster_orders(input_ids)

        labels = input_ids.clone()
        # prepend condition token
        input_ids = torch.cat([condition.view(condition.shape[0], -1),
                               input_ids.view(input_ids.shape[0], -1),], dim=1)
        embeddings = self.embeddings(input_ids)
        condition_token = embeddings[:, 0]

        # prepare positional embeddings.
        pos_embed = self.pos_embed.repeat(input_ids.shape[0], 1, 1)
        # cls_token, condition, the permute does not impact these prefix tokens.
        prefix = 2
        pos_embed_prefix = pos_embed[:, :prefix]
        # shuffle pos embed
        # if orders.shape[1] > 0:
        pos_embed_postfix = self.shuffle(pos_embed[:, prefix:prefix+self.image_seq_len], orders)
        if not is_sampling:
            # No need to shuffle labels for dAR
            # labels = self.shuffle(labels, orders)
            # randomized permutation: shuffle the embeddings of image tokens
            embeddings = torch.cat([embeddings[:, :1], self.shuffle(embeddings[:, 1:], orders)], dim=1)

        x = embeddings
        # prepend the cls token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) # Now each item in x is [cls_token, condition, image_tokens]

        # add original pos embed
        x = x + torch.cat([pos_embed_prefix, pos_embed_postfix], dim=1)[:, :x.shape[1]]

        # causal attention masking
        attn_mask = self.attn_mask[:x.shape[1], :x.shape[1]]
        
        # seperate condition token for each step, at generation, we start from 1 to seq len
        condition_token = condition_token.unsqueeze(1) + self.timesteps_embeddings[:, :x.shape[1]]
        if self.blocks[0].attn.kv_cache:
            if self.blocks[0].attn.k_cache is not None and self.blocks[0].attn.v_cache is not None:
                # only need to process the last token
                start_idx = self.blocks[0].attn.k_cache.shape[-2]
                x = x[:, start_idx:]
                attn_mask = attn_mask[-x.shape[1]:, :]
                # attn_mask = None
                # only keep the last condition
                condition_token = condition_token[:, start_idx:]

        for idx, blk in enumerate(self.blocks):
            if self.use_checkpoint:
                x = torch.utils.checkpoint.checkpoint(
                        blk.forward, x, attn_mask, condition_token, use_reentrant=False)
            else:
                x = blk(x, attn_mask=attn_mask, c=condition_token)

        if not self.blocks[0].attn.kv_cache:
            # remove cls token
            x = x[:, prefix - 1:]
            condition_token = condition_token[:, prefix - 1:]
        x = self.adaln_before_head(x, condition_token)
        x = self.lm_head(x) # [B, image_seq_len, vocab_size * image_seq_len]
        x = x.reshape(x.shape[0], x.shape[1], self.image_seq_len, self.target_codebook_size)

        if return_labels:
            return x, labels, orders
        return x
    
    # Add gumbel noise
    def log(self, t, eps=1e-20):
        return torch.log(t.clamp(min=eps))
    def gumbel_noise(self, t):
        noise = torch.zeros_like(t).uniform_(0, 1)
        return -self.log(-self.log(noise))
    def add_gumbel_noise(self, t, temperature):
        return t + temperature * self.gumbel_noise(t)
    
    @torch.no_grad()
    def generate(self,
                 condition,
                 guidance_scale,
                 randomize_temperature,
                 guidance_scale_pow,
                 kv_cache=False,
                 num_sample_steps=8,
                 guidance_decay="linear",
                 **kwargs):
        condition = self.preprocess_condition(
            condition, cond_drop_prob=0.0)
        device = condition.device
        num_samples = condition.shape[0]
        ids = torch.full((num_samples, 0), -1, device=device)
        cfg_scale = 0.

        if kv_cache:
            self.enable_kv_cache()

        if kwargs.get("fix_orders", False):
            self.random_ratio = 0.0
        else:
            self.random_ratio = 1.0

        orders = self.sample_orders(ids)
        # print("orders:", orders)
        token_len = 0
        for step in range(num_sample_steps):
            ratio = 1. * (step + 1) / num_sample_steps
            annealed_temp = randomize_temperature * (1.0 - ratio)
            # token_ratio = 1 - np.arccos(ratio) / (math.pi * 0.5)
            token_ratio = ratio
            
            if guidance_decay == "power-cosine":
                guidance_scale_pow = torch.ones((1), device=device) * guidance_scale_pow
                scale_step = (1 - torch.cos(((step / num_sample_steps) ** guidance_scale_pow) * torch.pi)) * 1/2
                cfg_scale = (guidance_scale - 1) * scale_step + 1
            elif guidance_decay == "linear":
                cfg_scale = ratio * guidance_scale
            elif guidance_decay == "constant":
                cfg_scale = guidance_scale
            
            current_orders = orders[:, :token_len]
            if guidance_scale != 0:
                logits = self.forward_fn(
                    torch.cat([ids, ids], dim=0),
                    torch.cat([condition, self.get_none_condition(condition)], dim=0),
                    orders=torch.cat([current_orders, current_orders], dim=0),
                    is_sampling=True)
                cond_logits, uncond_logits = logits[:num_samples], logits[num_samples:]
                logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale
            else:
                logits = self.forward_fn(
                    ids, condition, orders=current_orders, is_sampling=True
                ) # [B, len(ids), image_seq_len, vocab_size]
            
            logits = logits[:, -1] # [B, image_seq_len, vocab_size]
            token_len = int(self.image_seq_len * token_ratio)
            current_orders = orders[:, :token_len]
            next_token_indices = orders[:, ids.shape[1]:token_len]
            batch_size = logits.shape[0]
            batch_indices = torch.arange(batch_size, device=device).unsqueeze(1)
            batch_indices = batch_indices.expand(-1, next_token_indices.shape[1])
            logits = logits[batch_indices, next_token_indices] # [B, N_t, vocab_size]

            sampled_ids = self.add_gumbel_noise(logits, annealed_temp).argmax(dim=-1)
            sampled = sampled_ids.reshape(ids.shape[0], -1) # [B, N_t]
            ids = torch.cat((ids, sampled), dim = -1)
        if not kwargs.get("fix_orders", False):
            ids = self.unshuffle(ids, orders)
        self.disable_kv_cache()
        return ids
    
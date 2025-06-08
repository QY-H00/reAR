"""This file contains implementation for MaskGIT model.

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
    https://github.com/huggingface/open-muse
    https://github.com/baaivision/MUSE-Pytorch
    https://github.com/sail-sg/MDT/blob/main/masked_diffusion/models.py
"""

import torch
from torch import nn
import numpy as np
import math
import torch.utils.checkpoint
from transformers import BertConfig, BertModel
from einops import rearrange

import json
from huggingface_hub import PyTorchModelHubMixin
from omegaconf import OmegaConf
from pathlib import Path

from modeling.modules.base_model import BaseModel
from modeling.modules.blocks import UViTBlock


class ImageBert(BaseModel, PyTorchModelHubMixin, tags=["arxiv:2406.07550", "image-generation"], repo_url="https://github.com/bytedance/1d-tokenizer", license="apache-2.0"):
    def __init__(self, config):

        if isinstance(config, dict):
            config = OmegaConf.create(config)

        super().__init__()
        self.config = config
        self.target_codebook_size = config.model.vq_model.codebook_size
        self.condition_num_classes = config.model.generator.condition_num_classes
        self.image_seq_len = config.model.generator.image_seq_len
        self.mask_token_id = self.target_codebook_size
        self.hidden_size = config.model.generator.hidden_size
        self.num_hidden_layers = config.model.generator.num_hidden_layers
        self.num_attention_heads = config.model.generator.num_attention_heads
        self.intermediate_size = config.model.generator.intermediate_size

        self.model = BertModel(BertConfig(
            vocab_size=self.target_codebook_size + self.condition_num_classes + 2,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act='gelu',
            hidden_dropout_prob=config.model.generator.dropout,
            attention_probs_dropout_prob=config.model.generator.attn_drop,
            max_position_embeddings=config.model.generator.image_seq_len + 1,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            pad_token_id=None,
            position_embedding_type="absolute",
            use_cache=True
        ), add_pooling_layer=False)
        self.model.lm_head = nn.Linear(self.hidden_size, self.target_codebook_size, bias=True)
        
        self.model.post_init()

    def _save_pretrained(self, save_directory: Path) -> None:
        """Save weights and config to a local directory."""
        # Assume 'self.config' is your DictConfig object
        # Convert to a regular dictionary
        dict_config = OmegaConf.to_container(self.config)
        # Save as JSON
        file_path = Path(save_directory) / "config.json"
        with open(file_path, 'w') as json_file:
            json.dump(dict_config, json_file, indent=4)
        super()._save_pretrained(save_directory)
    
    def forward(self, input_ids=None, condition=None, cond_drop_prob=0.1):
        # Token space:
        #  [0, codebook_size - 1]                       : those are the learned quantized image tokens
        #  codebook_size                                : the mask token used to mask image tokens
        #  [codebook_size + 1, codebook_size + nclass]  : the imagenet class tokens
        #  codebook_size + 1 + nclass                   : the class drop label
        drop_label_mask = torch.rand_like(condition, dtype=torch.float) < cond_drop_prob
        # Shift the classes
        condition = condition + self.target_codebook_size + 1  # [0, 999] -> [codebook_size + 1, codebook_size + 999]
        condition[drop_label_mask] = self.condition_num_classes + self.target_codebook_size + 1
        # prepend condition token
        if input_ids is not None:
            input_ids = torch.cat([condition.view(condition.shape[0], -1),
                                   input_ids.view(input_ids.shape[0], -1),], dim=1)
        else:
            # at least there should be masked token
            raise NotImplementedError
        model_output = self.model(input_ids=input_ids)
        model_output = model_output[0]
        return self.model.lm_head(model_output[:, 1:]) # remove cond
    
    def get_none_condition(self,
                           condition
                           ):
        # In forward it will be added with self.target_codebook_size + 1
        return torch.full_like(condition, self.condition_num_classes)
    
    @torch.no_grad()
    def diffusion_reverse(self,
                          condition,
                          guidance_scale=3.0,
                          guidance_scale_pow=3.0,
                          num_sample_steps=8,
                          randomize_temperature=0.1,
                          guidance_decay="power-cosine",
                          softmax_temperature_annealing=False,
                          eps=1e-5,
                          **kwargs):
        solver = self.config.model.generator.get("solver", "ddpm")
        if solver == "dpm_solver":
            solver = self._dpm_solver_update
        elif solver == "ddpm":
            solver = self._ddpm_update
        else:
            raise ValueError(f"Unsupported solver {solver}")
        device = condition.device
        num_samples = condition.shape[0]
        ids = torch.full((num_samples, self.image_seq_len), self.mask_token_id, device=device)

        num_timesteps = num_sample_steps
        timesteps = torch.linspace(1, eps, num_timesteps + 1, device=condition.device)
        dt = (1 - eps) / num_timesteps

        cfg_scale = guidance_scale if guidance_decay == "constant" else 0.

        for i in range(num_timesteps):
            # ref: https://github.com/kuleshov-group/mdlm/blob/master/diffusion.py#L658
            ratio = 1. * (i + 1) / num_sample_steps
            # annealed_temp = randomize_temperature * (1.0 - ratio)
            if softmax_temperature_annealing:
                # init_temp = 0.5
                # annealed_temp = init_temp + (randomize_temperature - init_temp) * (1 - ratio)
                annealed_temp = randomize_temperature * (1 - ratio) ** 2
            else:
                annealed_temp = randomize_temperature
            t = timesteps[i] * torch.ones(num_samples, 1, device=condition.device)
            if guidance_decay == "power-cosine":
                # ref: https://github.com/sail-sg/MDT/blob/main/masked_diffusion/models.py#L501
                guidance_scale_pow = torch.ones((1), device=device) * guidance_scale_pow
                scale_step = (1 - torch.cos(((i / num_sample_steps) ** guidance_scale_pow) * torch.pi)) * 1/2
                cfg_scale = (guidance_scale - 1) * scale_step + 1

            if cfg_scale != 0:
                updated_ids = solver(
                    torch.cat([ids, ids], dim=0),
                    torch.cat([condition, self.get_none_condition(condition)], dim=0),
                    torch.cat([t, t], dim=0), dt, cfg_scale, annealed_temp)
            else:
                updated_ids = solver(ids, condition, t, dt, cfg_scale=0, annealed_temp=annealed_temp)
            
            if self.config.model.generator.get("noregret", False):
                copy_flag = (ids != self.mask_token_id).to(ids.dtype)
                ids = copy_flag * ids + (1 - copy_flag) * updated_ids
            else:
                ids = updated_ids

            if guidance_decay == "linear":
                cfg_scale = ratio * guidance_scale

            if guidance_decay == "interval":
                cfg_scale = guidance_scale if ratio <= 0.3 or ratio >= 0.1 else 0.0

        # To ensure the decoder can decode something
        ids = torch.where(ids >= self.target_codebook_size, torch.floor(torch.rand_like(ids.float()) * self.target_codebook_size).long(), ids)
        return ids
    
    # Add gumbel noise
    def log(self, t, eps=1e-20):
        return torch.log(t.clamp(min=eps))
    
    def gumbel_noise(self, t):
        noise = torch.zeros_like(t).uniform_(0, 1)
        return -self.log(-self.log(noise))
    
    def add_gumbel_noise(self, t, temperature):
        return t + temperature * self.gumbel_noise(t)
    
    # ref: https://github.com/baaivision/MUSE-Pytorch/blob/master/libs/muse.py#L40
    @torch.no_grad()
    def generate(self,
                 condition,
                 guidance_scale=3.0,
                 guidance_decay="constant",
                 guidance_scale_pow=3.0,
                 randomize_temperature=4.5,
                 softmax_temperature_annealing=False,
                 num_sample_steps=8):
        if self.config.model.generator.is_diffusion:
            return self.diffusion_reverse(
                condition, 
                guidance_scale, 
                guidance_scale_pow, 
                num_sample_steps, 
                eps=1e-5, 
                randomize_temperature=randomize_temperature, 
                guidance_decay=guidance_decay, 
                softmax_temperature_annealing=softmax_temperature_annealing
            )
        if guidance_decay not in ["constant", "linear", "power-cosine", "interval"]:
            # contstant: constant guidance scale
            # linear: linear increasing the guidance scale as in MUSE
            # power-cosine: the guidance schedule from MDT
            raise ValueError(f"Unsupported guidance decay {guidance_decay}")
        device = condition.device
        ids = torch.full((condition.shape[0], self.image_seq_len),
                          self.mask_token_id, device=device)

        cfg_scale = guidance_scale if guidance_decay == "constant" else 0.

        for step in range(num_sample_steps):
            ratio = 1. * (step + 1) / num_sample_steps
            annealed_temp = randomize_temperature * (1.0 - ratio)
            is_mask = (ids == self.mask_token_id)

            if guidance_decay == "power-cosine":
                # ref: https://github.com/sail-sg/MDT/blob/main/masked_diffusion/models.py#L501
                guidance_scale_pow = torch.ones((1), device=device) * guidance_scale_pow
                scale_step = (1 - torch.cos(((step / num_sample_steps) ** guidance_scale_pow) * torch.pi)) * 1/2
                cfg_scale = (guidance_scale - 1) * scale_step + 1

            if cfg_scale != 0:
                cond_logits = self.forward(
                    ids, condition, cond_drop_prob=0.0
                )
                uncond_logits = self.forward(
                    ids, condition, cond_drop_prob=1.0
                )
                if guidance_decay == "power-cosine":
                    logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale
                else:
                    logits = cond_logits + (cond_logits - uncond_logits) * cfg_scale
            else:
                logits = self.forward(
                    ids, condition, cond_drop_prob=0.0
                )

            if softmax_temperature_annealing:
                softmax_temperature = 0.5 + 0.8 * (1 - ratio)
                logits = logits / softmax_temperature

            # Add gumbel noise
            def log(t, eps=1e-20):
                return torch.log(t.clamp(min=eps))
            def gumbel_noise(t):
                noise = torch.zeros_like(t).uniform_(0, 1)
                return -log(-log(noise))
            def add_gumbel_noise(t, temperature):
                return t + temperature * gumbel_noise(t)

            sampled_ids = add_gumbel_noise(logits, annealed_temp).argmax(dim=-1)
            sampled_logits = torch.squeeze(
                torch.gather(logits, dim=-1, index=torch.unsqueeze(sampled_ids, -1)), -1)
            sampled_ids = torch.where(is_mask, sampled_ids, ids)
            sampled_logits = torch.where(is_mask, sampled_logits, +np.inf).float()
            # masking
            mask_ratio = np.arccos(ratio) / (math.pi * 0.5)

            mask_len = torch.Tensor([np.floor(self.image_seq_len * mask_ratio)]).to(device)
            mask_len = torch.maximum(torch.Tensor([1]).to(device),
                                     torch.minimum(torch.sum(is_mask, dim=-1, keepdims=True) - 1,
                                                   mask_len))[0].squeeze()
            confidence = add_gumbel_noise(sampled_logits, annealed_temp)
            sorted_confidence, _ = torch.sort(confidence, axis=-1)
            cut_off = sorted_confidence[:, mask_len.long() - 1:mask_len.long()]
            masking = (confidence <= cut_off)
            if step == num_sample_steps - 1:
                ids = sampled_ids
            else:
                ids = torch.where(masking, self.mask_token_id, sampled_ids)

            if guidance_decay == "linear":
                cfg_scale = ratio * guidance_scale

            if guidance_decay == "interval":
                cfg_scale = guidance_scale if (ratio >= 0.5 or ratio <= 0.85) else 0.0
        return ids

    def masking_input_tokens(self, input_tokens):
        batch_size, seq_len = input_tokens.shape
        device = input_tokens.device

        timesteps = torch.zeros((batch_size,), device=device).float().uniform_(0, 1.0)
        mask_ratio = torch.acos(timesteps) / (math.pi * 0.5) # arccos schedule
        mask_ratio = torch.clamp(mask_ratio, min=1e-3, max=1. - 1e-3)
        num_token_masked = (seq_len * mask_ratio).round().clamp(min=1)
        batch_randperm = torch.rand(batch_size, seq_len, device=device).argsort(dim=-1)
        masks = batch_randperm < rearrange(num_token_masked, 'b -> b 1')
        masked_tokens = torch.where(masks, self.mask_token_id, input_tokens)
        return masked_tokens, masks, mask_ratio
    
    def get_mask_ratio(self, t):
        '''
        t is a tensor of shape (batch_size,), the timestep of the diffusion process
        '''
        mask_ratio = torch.acos(t) / (math.pi * 0.5) # arccos schedule
        mask_ratio = torch.clamp(mask_ratio, min=1e-3, max=1. - 1e-3)
        return mask_ratio
    
    def _sample_categorical(self, categorical_probs):
        gumbel_norm = (
            1e-10
            - (torch.rand_like(categorical_probs) + 1e-10).log())
        return (categorical_probs / (gumbel_norm)).argmax(dim=-1)
    
    def _ddpm_update(self, x, condition, t, dt, cfg_scale=0, annealed_temp=0):
        # ref: https://github.com/kuleshov-group/mdlm/blob/master/diffusion.py#L612
        alpha_t = 1 - self.get_mask_ratio(t)
        alpha_s = 1 - self.get_mask_ratio(t - dt)
        if alpha_t.ndim > 1:
            alpha_t = alpha_t.squeeze(-1)
        if alpha_s.ndim > 1:
            alpha_s = alpha_s.squeeze(-1)
        assert alpha_t.ndim == 1, alpha_t.shape
        assert alpha_s.ndim == 1, alpha_s.shape
        alpha_t = alpha_t[:, None, None]
        alpha_s = alpha_s[:, None, None]
        if cfg_scale > 0:
            # Classifier-free guidance
            logits = self.forward(x, condition, cond_drop_prob=0.0)
            num_samples = x.shape[0] // 2
            logits_cond = logits[:num_samples]
            logits_uncond = logits[num_samples:]
            logits = logits_cond + cfg_scale * (logits_cond - logits_uncond)
            alpha_t = alpha_t[:num_samples]
            alpha_s = alpha_s[:num_samples]
        else:
            logits = self.forward(x, condition, cond_drop_prob=0.0)

        assert alpha_t.ndim == logits.ndim
        # logits = self.add_gumbel_noise(logits, annealed_temp) # only annealing the sampling of unmasked tokens
        logits = logits / (annealed_temp + 1e-8)
        q_x0 = logits.softmax(dim=-1)
        q_xs_real_tokens = q_x0 * (1 - alpha_s) # alpha_t - alpha_s?
        q_xs_mask_token = alpha_s.expand(q_xs_real_tokens.shape[0], q_xs_real_tokens.shape[1], 1)
        q_xs = torch.cat([q_xs_real_tokens, q_xs_mask_token], dim=-1)
        xs = self._sample_categorical(q_xs)
        return xs


class UViTBert(ImageBert):
    def __init__(self, config):
        super().__init__(config=config)

        del self.model

        self.embeddings = nn.Embedding(
            self.target_codebook_size + self.condition_num_classes + 2,
            self.hidden_size)
        
        self.pos_embed = nn.init.trunc_normal_(
            nn.Parameter(torch.zeros(1, self.config.model.generator.image_seq_len + 1, self.hidden_size)), 0., 0.02)
 
        self.in_blocks = nn.ModuleList([
            UViTBlock(
                dim=self.hidden_size, num_heads=self.num_attention_heads, mlp_ratio=(self.intermediate_size / self.hidden_size),
                qkv_bias=False, qk_scale=None, norm_layer=nn.LayerNorm, use_checkpoint=False)
            for _ in range(self.num_hidden_layers // 2)])

        self.mid_block = UViTBlock(
                dim=self.hidden_size, num_heads=self.num_attention_heads, mlp_ratio=(self.intermediate_size / self.hidden_size),
                qkv_bias=False, qk_scale=None, norm_layer=nn.LayerNorm, use_checkpoint=False)

        self.out_blocks = nn.ModuleList([
            UViTBlock(
                dim=self.hidden_size, num_heads=self.num_attention_heads, mlp_ratio=(self.intermediate_size / self.hidden_size),
                qkv_bias=False, qk_scale=None, norm_layer=nn.LayerNorm, skip=True, use_checkpoint=False)
            for _ in range(self.num_hidden_layers // 2)])

        self.norm = nn.LayerNorm(self.hidden_size)
        self.lm_head = nn.Linear(self.hidden_size,
                                 self.target_codebook_size, bias=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Embedding):
            m.weight.data = nn.init.trunc_normal_(m.weight.data, mean=0.0, std=0.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, input_ids=None, condition=None, cond_drop_prob=0.1):
        # Token space:
        #  [0, codebook_size - 1]                       : those are the learned quantized image tokens
        #  codebook_size                                : the mask token used to mask image tokens
        #  [codebook_size + 1, codebook_size + nclass]  : the imagenet class tokens
        #  codebook_size + 1 + nclass                   : the class drop label
        drop_label_mask = torch.rand_like(condition, dtype=torch.float) < cond_drop_prob
        # Shift the classes
        condition = condition + self.target_codebook_size + 1  # [0, 999] -> [codebook_size + 1, codebook_size + 999]
        condition[drop_label_mask] = self.condition_num_classes + self.target_codebook_size + 1
        # prepend condition token
        if input_ids is not None:
            input_ids = torch.cat([condition.view(condition.shape[0], -1),
                                   input_ids.view(input_ids.shape[0], -1),], dim=1)
        else:
            # at least there should be masked token
            raise NotImplementedError
        # UViT forward
        embeddings = self.embeddings(input_ids)
        x = embeddings + self.pos_embed[:, :embeddings.shape[1]]
        skips = []
        for blk in self.in_blocks:
            x = blk(x)
            skips.append(x)
        x = self.mid_block(x)
        for blk in self.out_blocks:
            x = blk(x, skips.pop())
        x = self.norm(x)
        return self.lm_head(x[:, 1:]) # remove cond
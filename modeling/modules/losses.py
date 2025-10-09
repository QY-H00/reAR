"""Adapted from:
    https://github.com/bytedance/1d-tokenizer/blob/main/modeling/modules/losses.py
"""
from typing import Mapping, Text, Tuple
import torch

class ARLoss(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.target_vocab_size = config.model.vq_model.codebook_size
        self.criterion = torch.nn.CrossEntropyLoss(reduction="mean")
        self.transition_alignment = config.model.generator.get("transition_alignment", False)
        self.transition_alignment_weight = config.model.generator.get("transition_alignment_weight", 1.0)
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor, encode_z: torch.Tensor, decode_z: torch.Tensor, features: torch.Tensor, perturb_mask: torch.Tensor) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        shift_logits = logits[..., :-1, :].permute(0, 2, 1).contiguous() # NLC->NCL
        shift_labels = labels.contiguous()
        shift_logits = shift_logits.view(shift_logits.shape[0], self.target_vocab_size, -1)
        shift_labels = shift_labels.view(shift_labels.shape[0], -1)
        shift_labels = shift_labels.to(shift_logits.device)
        ar_loss = self.criterion(shift_logits, shift_labels)
        correct_tokens = (torch.argmax(shift_logits, dim=1) == shift_labels).sum(dim=1) / shift_labels.size(1)
        if self.transition_alignment:
            proj_loss = self.transition_alignment_loss(encode_z, decode_z, features, perturb_mask)
            loss = ar_loss + self.transition_alignment_weight * proj_loss
        else:
            proj_loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
            loss = ar_loss
        return loss, {"loss": loss.clone().detach(), "correct_tokens": correct_tokens.mean(), "ar_loss": ar_loss.detach(), "proj_loss": proj_loss.detach()}

    def mean_flat(self, x):
        """
        Take the mean over all non-batch dimensions.
        """
        return torch.mean(x, dim=list(range(1, len(x.size()))))

    def transition_alignment_loss(self, encode_z: torch.Tensor, decode_z: torch.Tensor, features: torch.Tensor, perturb_mask: torch.Tensor) -> torch.Tensor:
        # encoder_z: B, N+2, z_dim
        # decoder_z: B, N+2, z_dim
        # features: tokenizer quantized states, B, N, z_dim
        # perturb_mask: B, N
        assert not (encode_z is None and decode_z is None)
        
        # Prepare tensors
        if encode_z is not None:
            encode_z_patches = encode_z[:, 2:, :]  # B, N, z_dim
        else:
            encode_z_patches = decode_z[:, 1:-1, :]  # B, N, z_dim
            
        if decode_z is not None:
            decode_z_patches = decode_z[:, 1:-1, :]  # B, N, z_dim
        else:
            decode_z_patches = encode_z[:, 2:, :]  # B, N, z_dim
        
        # Prepare mask
        if perturb_mask is None:
            perturb_mask = torch.zeros_like(encode_z_patches[:, :, 0])
        encode_mask_float = 1.0 - perturb_mask.to(dtype=encode_z_patches.dtype)  # B, N
        
        # Normalize all tensors at once (vectorized)
        encode_z_patches = torch.nn.functional.normalize(encode_z_patches, dim=-1)  # B, N, z_dim
        decode_z_patches = torch.nn.functional.normalize(decode_z_patches, dim=-1)  # B, N, z_dim
        features = torch.nn.functional.normalize(features, dim=-1)  # B, N, z_dim
        
        # Compute dot products vectorized
        encode_dot = (encode_z_patches * features).sum(dim=-1)  # B, N
        decode_dot = (decode_z_patches * features).sum(dim=-1)  # B, N
        
        # Apply masking and compute losses
        encode_loss = torch.mean(-encode_dot * encode_mask_float)  # Scalar
        decode_loss = torch.mean(-decode_dot)  # Scalar
        
        proj_loss = encode_loss + decode_loss
        return proj_loss

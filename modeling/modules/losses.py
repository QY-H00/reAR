"""This files contains training loss implementation.

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

Ref:
    https://github.com/CompVis/taming-transformers/blob/master/taming/modules/losses/vqperceptual.py
"""
from typing import Mapping, Text, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.cuda.amp import autocast

from .perceptual_loss import PerceptualLoss
from .discriminator import NLayerDiscriminator


def hinge_d_loss(logits_real: torch.Tensor, logits_fake: torch.Tensor) -> torch.Tensor:
    """Hinge loss for discrminator.

    This function is borrowed from
    https://github.com/CompVis/taming-transformers/blob/master/taming/modules/losses/vqperceptual.py#L20
    """
    loss_real = torch.mean(F.relu(1.0 - logits_real))
    loss_fake = torch.mean(F.relu(1.0 + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def compute_lecam_loss(
    logits_real_mean: torch.Tensor,
    logits_fake_mean: torch.Tensor,
    ema_logits_real_mean: torch.Tensor,
    ema_logits_fake_mean: torch.Tensor
) -> torch.Tensor:
    """Computes the LeCam loss for the given average real and fake logits.

    Args:
        logits_real_mean -> torch.Tensor: The average real logits.
        logits_fake_mean -> torch.Tensor: The average fake logits.
        ema_logits_real_mean -> torch.Tensor: The EMA of the average real logits.
        ema_logits_fake_mean -> torch.Tensor: The EMA of the average fake logits.

    Returns:
        lecam_loss -> torch.Tensor: The LeCam loss.
    """
    lecam_loss = torch.mean(torch.pow(F.relu(logits_real_mean - ema_logits_fake_mean), 2))
    lecam_loss += torch.mean(torch.pow(F.relu(ema_logits_real_mean - logits_fake_mean), 2))
    return lecam_loss

def compute_mdlm_loss(
    model_output: torch.Tensor,
    xt: torch.Tensor,
    x0: torch.Tensor,
    t: torch.Tensor,
    mask_index: int = -1,
    T: int = 1000
) -> torch.Tensor:
    """
    Compute the Masked Diffusion Language Model Loss.
    Ref: https://github.com/kuleshov-group/mdlm/blob/master/diffusion.py#L329

    Args:
        model_output -> torch.Tensor: [B, S, V], The output logits of the model, [[[0.5, 0.4, 0.1, 0.1], [0.5, 0.4, 0.1, 0.1], ...], [[], [], [], []]]
        xt -> torch.Tensor: [B, S], Diffused (masked) token sequence, e.g.: [[C, [MASK], A, B], [[MASK], A, B, C]] .
        x0 -> torch.Tensor: [B, S], Clean token sequence, e.g.: [[C, A, A, B], [B, A, B, C]] .
        t -> torch.Tensor: [B,], The timestep tensor.
        mask_index -> int: The mask index, we use the last token in the codebook to represent the mask token.
        T -> int: The number of discretized timesteps, commonly set as 1000.

    Returns:
        mdlm_loss -> torch.Tensor: The Masked Diffusion Language Model Loss.
    """
    dt = 1 / T

    if torch.is_tensor(t):
      t = t[:, None]
      assert t.ndim == 2
      t = t.clamp(0., 1. - 1e-4)
    alpha_t = 1 - t + torch.zeros_like(xt)
    alpha_s = 1 - (t - dt) + torch.zeros_like(xt)

    log_x_theta_at_x0 = torch.gather(
      model_output, -1, x0[:, :, None]).squeeze(-1)
    log_x_theta_at_m = model_output[:, :, mask_index]
    x_theta_at_m = log_x_theta_at_m.exp()
    
    term_1_coef = dt / t
    term_1_log_nr = torch.log(alpha_t * x_theta_at_m / t + 1)
    term_1_log_dr = log_x_theta_at_x0
    
    term_2_coef = 1 - dt / t
    term_2_log_nr = term_1_log_nr
    term_2_log_dr = torch.log(alpha_s * x_theta_at_m / (t - dt) + 1)

    L_vb_masked = (
      term_1_coef * (term_1_log_nr - term_1_log_dr)
      + term_2_coef * (term_2_log_nr - term_2_log_dr))

    L_vb = L_vb_masked * (xt == mask_index)
    return T * L_vb


class ReconstructionLoss_Stage1(torch.nn.Module):
    def __init__(
        self,
        config
    ):
        super().__init__()
        loss_config = config.losses
        self.quantizer_weight = loss_config.quantizer_weight
        self.target_codebook_size = 1024

    def forward(self,
                target_codes: torch.Tensor,
                reconstructions: torch.Tensor,
                quantizer_loss: torch.Tensor,
                ) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        return self._forward_generator(target_codes, reconstructions, quantizer_loss)

    def _forward_generator(self,
                           target_codes: torch.Tensor,
                           reconstructions: torch.Tensor,
                           quantizer_loss: Mapping[Text, torch.Tensor],
                           ) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        reconstructions = reconstructions.contiguous()
        loss_fct = nn.CrossEntropyLoss(reduction="mean")
        batch_size = reconstructions.shape[0]
        reconstruction_loss = loss_fct(reconstructions.view(batch_size, self.target_codebook_size, -1),
                                        target_codes.view(batch_size, -1))
        total_loss = reconstruction_loss + \
            self.quantizer_weight * quantizer_loss["quantizer_loss"]

        loss_dict = dict(
            total_loss=total_loss.clone().detach(),
            reconstruction_loss=reconstruction_loss.detach(),
            quantizer_loss=(self.quantizer_weight * quantizer_loss["quantizer_loss"]).detach(),
            commitment_loss=quantizer_loss["commitment_loss"].detach(),
            codebook_loss=quantizer_loss["codebook_loss"].detach(),
        )

        return total_loss, loss_dict


class ReconstructionLoss_Stage2(torch.nn.Module):
    def __init__(
        self,
        config
    ):
        """Initializes the losses module.

        Args:
            config: A dictionary, the configuration for the model and everything else.
        """
        super().__init__()
        loss_config = config.losses
        self.discriminator = NLayerDiscriminator()

        self.reconstruction_loss = loss_config.reconstruction_loss
        self.reconstruction_weight = loss_config.reconstruction_weight
        self.quantizer_weight = loss_config.quantizer_weight
        self.perceptual_loss = PerceptualLoss(
            loss_config.perceptual_loss).eval()
        self.perceptual_weight = loss_config.perceptual_weight
        self.discriminator_iter_start = loss_config.discriminator_start

        self.discriminator_factor = loss_config.discriminator_factor
        self.discriminator_weight = loss_config.discriminator_weight
        self.lecam_regularization_weight = loss_config.lecam_regularization_weight
        self.lecam_ema_decay = loss_config.get("lecam_ema_decay", 0.999)
        if self.lecam_regularization_weight > 0.0:
            self.register_buffer("ema_real_logits_mean", torch.zeros((1)))
            self.register_buffer("ema_fake_logits_mean", torch.zeros((1)))

        self.config = config

    @autocast(enabled=False)
    def forward(self,
                inputs: torch.Tensor,
                reconstructions: torch.Tensor,
                extra_result_dict: Mapping[Text, torch.Tensor],
                global_step: int,
                mode: str = "generator",
                ) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        # Both inputs and reconstructions are in range [0, 1].
        inputs = inputs.float()
        reconstructions = reconstructions.float()

        if mode == "generator":
            return self._forward_generator(inputs, reconstructions, extra_result_dict, global_step)
        elif mode == "discriminator":
            return self._forward_discriminator(inputs, reconstructions, global_step)
        else:
            raise ValueError(f"Unsupported mode {mode}")
   
    def should_discriminator_be_trained(self, global_step : int):
        return global_step >= self.discriminator_iter_start

    def _forward_generator(self,
                           inputs: torch.Tensor,
                           reconstructions: torch.Tensor,
                           extra_result_dict: Mapping[Text, torch.Tensor],
                           global_step: int
                           ) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        """Generator training step."""
        inputs = inputs.contiguous()
        reconstructions = reconstructions.contiguous()
        if self.reconstruction_loss == "l1":
            reconstruction_loss = F.l1_loss(inputs, reconstructions, reduction="mean")
        elif self.reconstruction_loss == "l2":
            reconstruction_loss = F.mse_loss(inputs, reconstructions, reduction="mean")
        else:
            raise ValueError(f"Unsuppored reconstruction_loss {self.reconstruction_loss}")
        reconstruction_loss *= self.reconstruction_weight

        # Compute perceptual loss.
        perceptual_loss = self.perceptual_loss(inputs, reconstructions).mean()

        # Compute discriminator loss.
        generator_loss = torch.zeros((), device=inputs.device)
        discriminator_factor = self.discriminator_factor if self.should_discriminator_be_trained(global_step) else 0
        d_weight = 1.0
        if discriminator_factor > 0.0 and self.discriminator_weight > 0.0:
            # Disable discriminator gradients.
            for param in self.discriminator.parameters():
                param.requires_grad = False
            logits_fake = self.discriminator(reconstructions)
            generator_loss = -torch.mean(logits_fake)

        d_weight *= self.discriminator_weight

        # Compute quantizer loss.
        quantizer_loss = extra_result_dict["quantizer_loss"]
        total_loss = (
            reconstruction_loss
            + self.perceptual_weight * perceptual_loss
            + self.quantizer_weight * quantizer_loss
            + d_weight * discriminator_factor * generator_loss
        )
        loss_dict = dict(
            total_loss=total_loss.clone().detach(),
            reconstruction_loss=reconstruction_loss.detach(),
            perceptual_loss=(self.perceptual_weight * perceptual_loss).detach(),
            quantizer_loss=(self.quantizer_weight * quantizer_loss).detach(),
            weighted_gan_loss=(d_weight * discriminator_factor * generator_loss).detach(),
            discriminator_factor=torch.tensor(discriminator_factor),
            commitment_loss=extra_result_dict["commitment_loss"].detach(),
            codebook_loss=extra_result_dict["codebook_loss"].detach(),
            d_weight=d_weight,
            gan_loss=generator_loss.detach(),
        )

        return total_loss, loss_dict

    def _forward_discriminator(self,
                               inputs: torch.Tensor,
                               reconstructions: torch.Tensor,
                               global_step: int,
                               ) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        """Discrminator training step."""
        discriminator_factor = self.discriminator_factor if self.should_discriminator_be_trained(global_step) else 0
        loss_dict = {}
        # Turn the gradients on.
        for param in self.discriminator.parameters():
            param.requires_grad = True

        real_images = inputs.detach().requires_grad_(True)
        logits_real = self.discriminator(real_images)
        logits_fake = self.discriminator(reconstructions.detach())

        discriminator_loss = discriminator_factor * hinge_d_loss(logits_real=logits_real, logits_fake=logits_fake)

        # optional lecam regularization
        lecam_loss = torch.zeros((), device=inputs.device)
        if self.lecam_regularization_weight > 0.0:
            lecam_loss = compute_lecam_loss(
                torch.mean(logits_real),
                torch.mean(logits_fake),
                self.ema_real_logits_mean,
                self.ema_fake_logits_mean
            ) * self.lecam_regularization_weight

            self.ema_real_logits_mean = self.ema_real_logits_mean * self.lecam_ema_decay + torch.mean(logits_real).detach()  * (1 - self.lecam_ema_decay)
            self.ema_fake_logits_mean = self.ema_fake_logits_mean * self.lecam_ema_decay + torch.mean(logits_fake).detach()  * (1 - self.lecam_ema_decay)
        
        discriminator_loss += lecam_loss

        loss_dict = dict(
            discriminator_loss=discriminator_loss.detach(),
            logits_real=logits_real.detach().mean(),
            logits_fake=logits_fake.detach().mean(),
            lecam_loss=lecam_loss.detach(),
        )
        return discriminator_loss, loss_dict


class ReconstructionLoss_Single_Stage(ReconstructionLoss_Stage2):
    def __init__(
        self,
        config
    ):
        super().__init__(config)
        loss_config = config.losses
        self.quantize_mode = config.model.vq_model.get("quantize_mode", "vq")
        
        if self.quantize_mode == "vae":
            self.kl_weight = loss_config.get("kl_weight", 1e-6)
            logvar_init = loss_config.get("logvar_init", 0.0)
            self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init, requires_grad=False)

    def _forward_generator(self,
                           inputs: torch.Tensor,
                           reconstructions: torch.Tensor,
                           extra_result_dict: Mapping[Text, torch.Tensor],
                           global_step: int
                           ) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        """Generator training step."""
        inputs = inputs.contiguous()
        reconstructions = reconstructions.contiguous()
        if self.reconstruction_loss == "l1":
            reconstruction_loss = F.l1_loss(inputs, reconstructions, reduction="mean")
        elif self.reconstruction_loss == "l2":
            reconstruction_loss = F.mse_loss(inputs, reconstructions, reduction="mean")
        else:
            raise ValueError(f"Unsuppored reconstruction_loss {self.reconstruction_loss}")
        reconstruction_loss *= self.reconstruction_weight

        # Compute perceptual loss.
        perceptual_loss = self.perceptual_loss(inputs, reconstructions).mean()

        # Compute discriminator loss.
        generator_loss = torch.zeros((), device=inputs.device)
        discriminator_factor = self.discriminator_factor if self.should_discriminator_be_trained(global_step) else 0
        d_weight = 1.0
        if discriminator_factor > 0.0 and self.discriminator_weight > 0.0:
            # Disable discriminator gradients.
            for param in self.discriminator.parameters():
                param.requires_grad = False
            logits_fake = self.discriminator(reconstructions)
            generator_loss = -torch.mean(logits_fake)

        d_weight *= self.discriminator_weight

        if self.quantize_mode == "vq":
            # Compute quantizer loss.
            quantizer_loss = extra_result_dict["quantizer_loss"]
            total_loss = (
                reconstruction_loss
                + self.perceptual_weight * perceptual_loss
                + self.quantizer_weight * quantizer_loss
                + d_weight * discriminator_factor * generator_loss
            )
            loss_dict = dict(
                total_loss=total_loss.clone().detach(),
                reconstruction_loss=reconstruction_loss.detach(),
                perceptual_loss=(self.perceptual_weight * perceptual_loss).detach(),
                quantizer_loss=(self.quantizer_weight * quantizer_loss).detach(),
                weighted_gan_loss=(d_weight * discriminator_factor * generator_loss).detach(),
                discriminator_factor=torch.tensor(discriminator_factor),
                commitment_loss=extra_result_dict["commitment_loss"].detach(),
                codebook_loss=extra_result_dict["codebook_loss"].detach(),
                d_weight=d_weight,
                gan_loss=generator_loss.detach(),
            )
        elif self.quantize_mode == "vae":
            # Compute kl loss.
            reconstruction_loss = reconstruction_loss / torch.exp(self.logvar)
            posteriors = extra_result_dict
            kl_loss = posteriors.kl()
            kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
            total_loss = (
                reconstruction_loss
                + self.perceptual_weight * perceptual_loss
                + self.kl_weight * kl_loss
                + d_weight * discriminator_factor * generator_loss
            )
            loss_dict = dict(
                total_loss=total_loss.clone().detach(),
                reconstruction_loss=reconstruction_loss.detach(),
                perceptual_loss=(self.perceptual_weight * perceptual_loss).detach(),
                kl_loss=(self.kl_weight * kl_loss).detach(),
                weighted_gan_loss=(d_weight * discriminator_factor * generator_loss).detach(),
                discriminator_factor=torch.tensor(discriminator_factor),
                d_weight=d_weight,
                gan_loss=generator_loss.detach(),
            )
        else:
            raise NotImplementedError

        return total_loss, loss_dict



class MLMLoss(torch.nn.Module):
    def __init__(self,
                 config):
        super().__init__()
        self.label_smoothing = config.losses.label_smoothing
        self.is_diffusion = config.losses.is_diffusion
        self.loss_weight_unmasked_token = config.losses.loss_weight_unmasked_token
        self.uniform_loss_weights = config.losses.get("uniform_loss_weights", False)
        self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=self.label_smoothing,
                                                   reduction="none")
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor,
                weights=None, mask_ratios=None) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        inputs = rearrange(inputs, "b n c -> b c n")
        loss = self.criterion(inputs, targets)
        weights = weights.to(loss)
        loss_weights = (1.0 - weights) * self.loss_weight_unmasked_token + weights # set 0 to self.loss_weight_unasked_token
        if self.is_diffusion:
            if self.uniform_loss_weights:
                loss_weights = torch.ones_like(loss_weights)
            diffusion_weights = 1 / mask_ratios
            diffusion_weights = diffusion_weights[:, None]
            loss_weights = loss_weights * diffusion_weights
        loss = (loss * loss_weights).sum() / (loss_weights.sum() + 1e-8)
        # we only compute correct tokens on masked tokens
        correct_tokens = ((torch.argmax(inputs, dim=1) == targets) * weights).sum(dim=1) / (weights.sum(1) + 1e-8)
        return loss, {"loss": loss, "correct_tokens": correct_tokens.mean()}
    

class ARLoss(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.target_vocab_size = config.model.vq_model.codebook_size
        self.criterion = torch.nn.CrossEntropyLoss(reduction="mean")
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        shift_logits = logits[..., :-1, :].permute(0, 2, 1).contiguous() # NLC->NCL
        shift_labels = labels.contiguous()
        shift_logits = shift_logits.view(shift_logits.shape[0], self.target_vocab_size, -1)
        shift_labels = shift_labels.view(shift_labels.shape[0], -1)
        shift_labels = shift_labels.to(shift_logits.device)
        loss = self.criterion(shift_logits, shift_labels)
        correct_tokens = (torch.argmax(shift_logits, dim=1) == shift_labels).sum(dim=1) / shift_labels.size(1)
        return loss, {"loss": loss, "correct_tokens": correct_tokens.mean()}
    

class dARLoss(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.target_vocab_size = config.model.vq_model.codebook_size
        self.no_mask = config.losses.get("no_mask", False)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="none")
        N = config.model.generator.image_seq_len
        self.valid_mask =  torch.triu(torch.ones(N, N), diagonal=0)
        self.batch_size = config.training.per_gpu_batch_size
        # self.valid_mask = torch.ones(N, N)
        row_indices = torch.arange(N).view(-1, 1)
        scaling_factors = 1.0 / (N - row_indices + 1e-8)  # Add small epsilon to avoid division by zero
        self.prediction_mask = self.valid_mask * scaling_factors
        self.valid_mask = self.valid_mask.unsqueeze(0).repeat(self.batch_size, 1, 1)
        self.prediction_mask = self.prediction_mask.unsqueeze(0).repeat(self.batch_size, 1, 1)
        # self.prediction_mask = self.prediction_mask * ((N + 1) * N / 2) / torch.sum(self.prediction_mask)

    def unshuffle_mask(self, mask, orders):
        # Unshuffle the tensor based on the original orders
        B, N, N = mask.shape # orders: [B, N]
        
        # Create inverse order using vectorized operations
        inverse_orders = torch.zeros_like(orders)
        batch_indices = torch.arange(N, device=orders.device).unsqueeze(0).expand(B, -1)
        inverse_orders.scatter_(1, orders, batch_indices)
        
        # Expand inverse orders to match mask dimensions for column-wise gathering
        inverse_orders_expanded = inverse_orders.unsqueeze(1).expand(-1, N, -1)  # [B, N, N]
        # Use torch.gather to reorder columns for all batches simultaneously
        unshuffled_mask = torch.gather(mask, 2, inverse_orders_expanded)
        return unshuffled_mask
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor, orders: torch.Tensor) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        # logits: B, image_seq_len, image_seq_len, vocab_size
        B, input_len, N, vocab_size = logits.shape
        shift_logits = logits[..., :-1, :, :].view(B, N ** 2, vocab_size).permute(0, 2, 1).contiguous() # (B, N+1, N, V) -> (B, V, N*N)
        shift_logits = shift_logits.view(shift_logits.shape[0], self.target_vocab_size, -1) # [B, vocab_size, N*N]
        shift_labels = labels.repeat(1, N).contiguous() # [B, N*N]
        shift_labels = shift_labels.view(shift_labels.shape[0], -1).to(shift_logits.device) # [B, N*N]
        loss = self.criterion(shift_logits, shift_labels) # [B, N*N]

        if not self.no_mask:
            mask = self.prediction_mask.to(loss.device)
            mask = self.unshuffle_mask(mask, orders)
            loss = loss.view(B, N, N)
            loss = loss * mask
            num_elements = mask.sum()
            loss = loss.sum() / num_elements
        else:
            loss = loss.mean()

        correct_matrix = (torch.argmax(shift_logits, dim=1) == shift_labels).view(B, N, N)
        valid_mask = self.valid_mask.to(correct_matrix.device)
        valid_mask = self.unshuffle_mask(valid_mask, orders)
        correct_matrix = correct_matrix * valid_mask
        valid_num_elements = valid_mask.sum()
        correct_tokens = correct_matrix.sum() / valid_num_elements

        return loss, {"loss": loss, "correct_tokens": correct_tokens}

class MDLMLoss(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.T = config.get("model.generator.T", 1000)
        self.mask_index = config.model.vq_model.codebook_size
        self.same_expectation = config.get("losses.same_expectation", False)
        if self.same_expectation:
            self.loss_weight = self.T / math.log(self.T) # This is to adjust the loss expectation on masked tokens as the same as MLMLoss
        else:
            self.loss_weight = 1.0
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor, extra_dict: Mapping[Text, torch.Tensor]) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        loss = compute_mdlm_loss(model_output=logits, xt=labels, x0=extra_dict["x0"], t=extra_dict["t"], mask_index=self.mask_index, T=self.T).mean()
        logits = rearrange(logits, "b n c -> b c n")
        mask = extra_dict["mask"]
        correct_tokens = ((torch.argmax(logits, dim=1) == extra_dict["x0"]) * mask).sum(dim=1) / (mask.sum(1) + 1e-8)
        return loss * self.loss_weight, {"loss": loss, "correct_tokens": correct_tokens.mean()}
    
class dMLMLoss(torch.nn.Module):
    def __init__(self,
                 config):
        super().__init__()
        self.T = config.get("model.generator.T", 1000)
        self.label_smoothing = config.losses.label_smoothing
        self.loss_weight_unmasked_token = config.losses.loss_weight_unmasked_token
        self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=self.label_smoothing,
                                                   reduction="none")
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor, extra_dict: Mapping[Text, torch.Tensor]) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        inputs = rearrange(logits, "b n c -> b c n")
        targets = extra_dict["x0"]
        loss = self.criterion(inputs, targets)
        weights = extra_dict["mask"].float()
        diffusion_weight = 1 / extra_dict["t"]
        diffusion_weight = diffusion_weight / diffusion_weight.detach().mean()
        diffusion_weight = diffusion_weight[:, None]
        assert diffusion_weight.ndim == weights.ndim
        loss_weights = (1.0 - weights) * self.loss_weight_unmasked_token + weights # set 0 to self.loss_weight_unasked_token
        loss_weights = loss_weights
        loss = (diffusion_weight * loss * loss_weights).sum() / (loss_weights.sum() + 1e-8)
        # we only compute correct tokens on masked tokens
        correct_tokens = ((torch.argmax(inputs, dim=1) == targets) * weights).sum(dim=1) / (weights.sum(1) + 1e-8)
        return loss, {"loss": loss, "correct_tokens": correct_tokens.mean()}

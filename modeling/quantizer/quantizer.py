"""Vector quantizer.

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
    https://github.com/CompVis/taming-transformers/blob/master/taming/modules/vqvae/quantize.py
    https://github.com/google-research/magvit/blob/main/videogvt/models/vqvae.py
    https://github.com/CompVis/latent-diffusion/blob/main/ldm/modules/distributions/distributions.py
    https://github.com/lyndonzheng/CVQ-VAE/blob/main/quantise.py
"""
from typing import Mapping, Text, Tuple

import torch
from einops import rearrange
from accelerate.utils.operations import gather
from torch.cuda.amp import autocast

from modeling.quantizer.quantizer_utils import entropy_loss_fn

class VectorQuantizer(torch.nn.Module):
    def __init__(self,
                 codebook_size: int = 1024,
                 token_size: int = 256,
                 commitment_cost: float = 0.25,
                 use_l2_norm: bool = False,
                 clustering_vq: bool = False
                 ):
        super().__init__()
        self.codebook_size = codebook_size
        self.token_size = token_size
        self.commitment_cost = commitment_cost

        self.embedding = torch.nn.Embedding(codebook_size, token_size)
        self.embedding.weight.data.uniform_(-1.0 / codebook_size, 1.0 / codebook_size)
        self.use_l2_norm = use_l2_norm

        self.clustering_vq = clustering_vq
        if clustering_vq:
            self.decay = 0.99
            self.register_buffer("embed_prob", torch.zeros(self.codebook_size))

    # Ensure quantization is performed using f32
    @autocast(enabled=False)
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        z = z.float()
        z = rearrange(z, 'b c h w -> b h w c').contiguous()
        z_flattened = rearrange(z, 'b h w c -> (b h w) c')
        unnormed_z_flattened = z_flattened

        if self.use_l2_norm:
            z_flattened = torch.nn.functional.normalize(z_flattened, dim=-1)
            embedding = torch.nn.functional.normalize(self.embedding.weight, dim=-1)
        else:
            embedding = self.embedding.weight
        d = torch.sum(z_flattened**2, dim=1, keepdim=True) + \
            torch.sum(embedding**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, embedding.T)

        min_encoding_indices = torch.argmin(d, dim=1) # num_ele
        z_quantized = self.get_codebook_entry(min_encoding_indices).view(z.shape)

        if self.use_l2_norm:
            z = torch.nn.functional.normalize(z, dim=-1)

        # compute loss for embedding
        commitment_loss = self.commitment_cost * torch.mean((z_quantized.detach() - z) **2)
        codebook_loss = torch.mean((z_quantized - z.detach()) **2)

        if self.clustering_vq and self.training:
            with torch.no_grad():
                # Gather distance matrix from all GPUs.
                encoding_indices = gather(min_encoding_indices)
                if len(min_encoding_indices.shape) != 1:
                    raise ValueError(f"min_encoding_indices in a wrong shape, {min_encoding_indices.shape}")
                # Compute and update the usage of each entry in the codebook.
                encodings = torch.zeros(encoding_indices.shape[0], self.codebook_size, device=z.device)
                encodings.scatter_(1, encoding_indices.unsqueeze(1), 1)
                avg_probs = torch.mean(encodings, dim=0)
                self.embed_prob.mul_(self.decay).add_(avg_probs, alpha=1-self.decay)
                # Closest sampling to update the codebook.
                all_d = gather(d)
                all_unnormed_z_flattened = gather(unnormed_z_flattened).detach()
                if all_d.shape[0] != all_unnormed_z_flattened.shape[0]:
                    raise ValueError(
                        "all_d and all_unnormed_z_flattened have different length" + 
                        f"{all_d.shape}, {all_unnormed_z_flattened.shape}")
                indices = torch.argmin(all_d, dim=0)
                random_feat = all_unnormed_z_flattened[indices]
                # Decay parameter based on the average usage.
                decay = torch.exp(-(self.embed_prob * self.codebook_size * 10) /
                                   (1 - self.decay) - 1e-3).unsqueeze(1).repeat(1, self.token_size)
                self.embedding.weight.data = self.embedding.weight.data * (1 - decay) + random_feat * decay

        loss = commitment_loss + codebook_loss

        # preserve gradients
        z_quantized = z + (z_quantized - z).detach()

        # reshape back to match original input shape
        z_quantized = rearrange(z_quantized, 'b h w c -> b c h w').contiguous()

        result_dict = dict(
            quantizer_loss=loss,
            commitment_loss=commitment_loss,
            codebook_loss=codebook_loss,
            min_encoding_indices=min_encoding_indices.view(z_quantized.shape[0], z_quantized.shape[2], z_quantized.shape[3])
        )

        return z_quantized, result_dict

    def get_codebook_entry(self, indices):
        if len(indices.shape) == 1:
            z_quantized = self.embedding(indices)
        elif len(indices.shape) == 2:
            z_quantized = torch.einsum('bd,dn->bn', indices, self.embedding.weight)
        else:
            raise NotImplementedError
        if self.use_l2_norm:
            z_quantized = torch.nn.functional.normalize(z_quantized, dim=-1)
        return z_quantized
    

class DiagonalGaussianDistribution(object):
    @autocast(enabled=False)
    def __init__(self, parameters, deterministic=False):
        """Initializes a Gaussian distribution instance given the parameters.

        Args:
            parameters (torch.Tensor): The parameters for the Gaussian distribution. It is expected
                to be in shape [B, 2 * C, *], where B is batch size, and C is the embedding dimension.
                First C channels are used for mean and last C are used for logvar in the Gaussian distribution.
            deterministic (bool): Whether to use deterministic sampling. When it is true, the sampling results
                is purely based on mean (i.e., std = 0).
        """
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters.float(), 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    @autocast(enabled=False)
    def sample(self):
        x = self.mean.float() + self.std.float() * torch.randn(self.mean.shape).to(device=self.parameters.device)
        return x

    @autocast(enabled=False)
    def mode(self):
        return self.mean

    @autocast(enabled=False)
    def kl(self):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            return 0.5 * torch.sum(torch.pow(self.mean.float(), 2)
                                    + self.var.float() - 1.0 - self.logvar.float(),
                                    dim=[1, 2])
        
class SimpleVectorizer(torch.nn.Module):
    def __init__(
        self,
        codebook_size: int = 1024,
        token_size: int = 256,
        commitment_cost: float = 0.25,
        entropy_loss_weight: float = 0.0,
        entropy_loss_temperature: float = 0.01,
        entropy_gamma: float = 1.0,
        use_l2_normalisation: bool = False,
    ):
        """ Initializes the quantizer.

        Args:
            codebook_size -> int: The size of the codebook.
            token_size -> int: The feature dimensions of the tokens.
            commitment_cost -> float: The commitment cost.
            entropy_loss_weight -> float: The weight of the entropy loss.
            entropy_loss_temperature -> float: The temperature of the entropy loss.
            entropy_gamma -> float: The gamma of the entropy loss.
            use_l2_normalisation -> bool: Whether to use L2 normalisation.
        """

        super().__init__()
        self.commitment_cost = commitment_cost

        self.embedding = torch.nn.Embedding(codebook_size, token_size)
        self.embedding.weight.data.uniform_(-1.0 / codebook_size, 1.0 / codebook_size)

        self.entropy_loss_weight = entropy_loss_weight
        self.entropy_loss_temperature = entropy_loss_temperature
        self.entropy_gamma = entropy_gamma
        self.use_l2_normalisation = use_l2_normalisation

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        """ Computes the quantization loss and returns the quantized latent representation.

        Args:
            z -> torch.Tensor: The latent representation.

        Returns:
            z_quantized -> torch.Tensor: The quantized latent representation.
            result_dict -> Mapping[Text, torch.Tensor]: A dictionary containing additional results
                and losses from the quantizer.
        """
        # reshape z -> (batch, height, width, channel) and flatten
        z = rearrange(z, 'b c h w -> b h w c').contiguous()

        if self.use_l2_normalisation:
            z = torch.nn.functional.normalize(z, dim=-1)
            embedding = torch.nn.functional.normalize(self.embedding.weight, dim=-1)
        else:
            embedding = self.embedding.weight

        z_flattened = rearrange(z, 'b h w c -> (b h w) c')

        # distances from z to embeddings e_j d = (z - e)^2 = z^2 + e^2 - 2 e * z
        d = torch.sum(z_flattened**2, dim=1, keepdim=True) + \
            torch.sum(embedding**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, embedding.T)

        min_encoding_indices = torch.argmin(d, dim=1)
        z_quantized = self.get_codebook_entry(min_encoding_indices).view(z.shape)

        # compute loss for embedding
        commitment_loss = self.commitment_cost * torch.mean((z_quantized.detach() - z) **2)
        codebook_loss = torch.mean((z_quantized - z.detach()) **2)
        entropy_loss = torch.zeros((), device=z.device)
        per_sample_entropy = torch.zeros((), device=z.device)
        avg_entropy = torch.zeros((), device=z.device)

        # Use entropy loss on the codebook
        if self.entropy_loss_weight != 0.0 and self.training:
            per_sample_entropy, avg_entropy = entropy_loss_fn(-1*d, self.entropy_loss_temperature, self.entropy_gamma)
            entropy_loss = self.entropy_loss_weight * (per_sample_entropy - avg_entropy)

        loss = commitment_loss + codebook_loss + entropy_loss

        # preserve gradients
        z_quantized = z + (z_quantized - z).detach()

        # reshape back to match original input shape
        z_quantized = rearrange(z_quantized, 'b h w c -> b c h w').contiguous()

        result_dict = dict(
            quantizer_loss=loss,
            commitment_loss=commitment_loss,
            codebook_loss=codebook_loss,
            entropy_loss=entropy_loss,
            per_sample_entropy=per_sample_entropy,
            avg_entropy=avg_entropy,
            min_encoding_indices=min_encoding_indices.view(z_quantized.shape[0], -1)
        )

        return z_quantized, result_dict

    def get_codebook_entry(self, indices: torch.Tensor) -> torch.Tensor:
        """ Returns the codebook entry for the given indices.
        
        Args:
            indices -> torch.Tensor: The indices of the codebook entries.

        Returns:
            z_quantized -> torch.Tensor: The codebook entries.
        """
        # get quantized latent vectors
        z_quantized = self.embedding(indices.int())
        if self.use_l2_normalisation:
            z_quantized = torch.nn.functional.normalize(z_quantized, dim=-1)
        return z_quantized

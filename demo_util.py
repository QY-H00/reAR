"""Demo file for sampling images from TiTok.

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
"""


import torch

from safetensors.torch import load_model
from omegaconf import OmegaConf
from modeling.titok import TiTok, MaskgitVQ
from modeling.maskgit import ImageBert, UViTBert
from modeling.rar import RAR
from modeling.dar import dAR
from modeling.qar import QAR
from modeling.vqplus import VQPlus # VQGAN+ From MaskBit


def get_config_cli():
    cli_conf = OmegaConf.from_cli()

    yaml_conf = OmegaConf.load(cli_conf.config)
    conf = OmegaConf.merge(yaml_conf, cli_conf)

    return conf

def get_config(config_path):
    conf = OmegaConf.load(config_path)
    return conf

def get_tokenizer(config):
    assert "model_type" in config.model.vq_model, "model_type is not specified in the config"
    if config.model.vq_model.model_type == "titok":
        return get_titok_tokenizer(config)
    elif config.model.vq_model.model_type == "gigatok":
        return get_gigatok_tokenizer(config)
    elif config.model.vq_model.model_type == "vqplus":
        return get_vqplus_tokenizer(config)
    elif config.model.vq_model.model_type == "maskgitvq":
        return get_maskgitvq_tokenizer(config)
    else:
        raise ValueError(f"Unsupported tokenizer type {config.model.tokenizer.model_type}")

def get_vqplus_tokenizer(config):
    # 2D Tokenizer from MaskBit
    tokenizer = VQPlus(config.model.vq_model)
    tokenizer.load_state_dict(torch.load(config.experiment.tokenizer_checkpoint, map_location="cpu"))
    tokenizer.eval()
    tokenizer.requires_grad_(False)
    return tokenizer

def get_titok_tokenizer(config):
    # 1D Tokenizer from TiTok
    if "yucornetto" in config.experiment.tokenizer_checkpoint:
        tokenizer = TiTok.from_pretrained(config.experiment.tokenizer_checkpoint)
    else:
        tokenizer = TiTok(config)
        tokenizer.load_state_dict(torch.load(config.experiment.tokenizer_checkpoint, map_location="cpu"))
    tokenizer.eval()
    tokenizer.requires_grad_(False)
    return tokenizer

def get_maskgitvq_tokenizer(config):
    tokenizer = MaskgitVQ(config.experiment.tokenizer_checkpoint)
    tokenizer.eval()
    tokenizer.requires_grad_(False)
    return tokenizer

def get_gigatok_tokenizer(config):
    raise NotImplementedError("GigaTok tokenizer is not implemented yet")

def get_titok_generator(config):
    if config.model.generator.model_type == "ViT":
        model_cls = ImageBert
    elif config.model.generator.model_type == "UViT":
        model_cls = UViTBert
    else:
        raise ValueError(f"Unsupported model type {config.model.generator.model_type}")
    if "yucornetto" in config.experiment.generator_checkpoint:
        generator = model_cls.from_pretrained(config.experiment.generator_checkpoint)
    else:
        generator = model_cls(config)
        generator.load_state_dict(torch.load(config.experiment.generator_checkpoint, map_location="cpu"))
    generator.eval()
    generator.requires_grad_(False)
    return generator

def get_rar_generator(config):
    model_cls = RAR
    generator = model_cls(config)
    if ".safetensors" in config.experiment.generator_checkpoint:
        missing, unexpected = load_model(generator, config.experiment.generator_checkpoint, device="cpu")
        print(missing)
        print(unexpected)
    else:
        generator.load_state_dict(torch.load(config.experiment.generator_checkpoint, map_location="cpu"))
    generator.eval()
    generator.requires_grad_(False)
    generator.set_random_ratio(0)
    return generator

def get_dar_generator(config):
    model_cls = dAR
    generator = model_cls(config)
    if ".safetensors" in config.experiment.generator_checkpoint:
        missing, unexpected = load_model(generator, config.experiment.generator_checkpoint, device="cpu")
        print(missing)
        print(unexpected)
    else:
        generator.load_state_dict(torch.load(config.experiment.generator_checkpoint, map_location="cpu"))
    generator.eval()
    generator.requires_grad_(False)
    return generator

def get_qar_generator(config):
    model_cls = QAR
    generator = model_cls(config)
    if ".safetensors" in config.experiment.generator_checkpoint:
        missing, unexpected = load_model(generator, config.experiment.generator_checkpoint, device="cpu")
        print(missing)
        print(unexpected)
    else:
        generator.load_state_dict(torch.load(config.experiment.generator_checkpoint, map_location="cpu"))
    generator.eval()
    generator.requires_grad_(False)
    return generator

@torch.no_grad()
def sample_fn(generator,
              tokenizer,
              labels=None,
              guidance_scale=3.0,
              guidance_decay="constant",
              guidance_scale_pow=3.0,
              randomize_temperature=2.0,
              softmax_temperature_annealing=False,
              num_sample_steps=8,
              device="cuda",
              return_tensor=False,
              kv_cache=False,
              fix_orders=False,
              use_annealed_temp=True,
              maskgit_sampling=False,
              top_k=None,
              top_p=None,
              inference_k_tokens=None):
    generator.eval()
    tokenizer.eval()
    if labels is None:
        # goldfish, chicken, tiger cat, hourglass, ship, dog, race car, airliner, teddy bear, random
        labels = [1, 7, 282, 604, 724, 179, 751, 404, 850, torch.randint(0, 999, size=(1,))]

    if not isinstance(labels, torch.Tensor):
        labels = torch.LongTensor(labels).to(device)

    generated_tokens = generator.generate(
        condition=labels,
        guidance_scale=guidance_scale,
        guidance_decay=guidance_decay,
        guidance_scale_pow=guidance_scale_pow,
        randomize_temperature=randomize_temperature,
        softmax_temperature_annealing=softmax_temperature_annealing,
        num_sample_steps=num_sample_steps,
        kv_cache=kv_cache,
        fix_orders=fix_orders,
        use_annealed_temp=use_annealed_temp,
        maskgit_sampling=maskgit_sampling,
        top_k=top_k,
        top_p=top_p,
        inference_k_tokens=inference_k_tokens
    )
    
    generated_image = tokenizer.decode_tokens(
        generated_tokens.view(generated_tokens.shape[0], -1)
    )
    if return_tensor:
        return generated_image

    generated_image = torch.clamp(generated_image, 0.0, 1.0)
    generated_image = (generated_image * 255.0).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

    return generated_image

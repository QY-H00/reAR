"""Adapted from:
    https://github.com/bytedance/1d-tokenizer/blob/main/demo_util.py
"""


import torch

from safetensors.torch import load_model
from omegaconf import OmegaConf
from modeling.titok import TiTok, MaskgitVQ
from modeling.rear import reAR
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

def get_rear_generator(config):
    model_cls = reAR
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

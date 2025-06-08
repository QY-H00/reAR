import torch
import os
from PIL import Image
import numpy as np
import demo_util
from huggingface_hub import hf_hub_download
from modeling.maskgit import ImageBert
from modeling.titok import TiTok
from data.imagenet_classes import imagenet_idx2classname

if __name__ == "__main__":
    is_diffusions = [False, True]
    no_regrets = [False]
    num_sample_stepss = [8, 16, 32, 64]
    solvers = ["ddpm"]
    save_dir = "assets/trial"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    titok_tokenizer = TiTok.from_pretrained("yucornetto/tokenizer_titok_b64_imagenet")
    titok_tokenizer.eval()
    titok_tokenizer.requires_grad_(False)

    for is_diffusion in is_diffusions:
        for noregret in no_regrets:
            for num_sample_steps in num_sample_stepss:
                for solver in solvers:
                    config = "/home/qiyuan/dGen/configs/training/generator/maskgit.yaml"
                    config = demo_util.get_config(config)
                    config.experiment.generator_checkpoint = "/home/qiyuan/dGen/temp/maskgit_scratch_titok_diffusion_check/checkpoint-800000/ema_model/pytorch_model.bin"
                    # config.experiment.generator_checkpoint = "/home/qiyuan/dGen/pretrained_checkpoints/generator_titok_b64_imagenet/model.safetensors"
                    # config.experiment.generator_checkpoint = "/home/qiyuan/dGen/temp/maskgit_dgen_posttrain_titok_dmlmloss_fix/checkpoint-800000/ema_model/pytorch_model.bin"
                    # config.experiment.generator_checkpoint = "yucornetto/generator_titok_b64_imagenet"
                    titok_generator = demo_util.get_titok_generator(config)
                    titok_generator.config.model.generator.is_diffusion = is_diffusion
                    titok_generator.config.model.generator.noregret = noregret
                    titok_generator.config.model.generator.solver = solver
                    titok_generator.eval()
                    titok_generator.requires_grad_(False)

                    device = "cuda"
                    titok_tokenizer = titok_tokenizer.to(device)
                    titok_generator = titok_generator.to(device)

                    # generate an image
                    torch.manual_seed(42)  # fixed seed for reproducibility
                    sample_labels = [torch.randint(0, 999, size=(1,)).item()] # random IN-1k class
                    # sample_labels = [207] # golden retriever
                    sample_labels = [284] # Siamese cat, Siamese
                    get_name = imagenet_idx2classname[sample_labels[0]].replace(" ", "_")
                    generated_image = demo_util.sample_fn(
                        generator=titok_generator,
                        tokenizer=titok_tokenizer,
                        labels=sample_labels,
                        guidance_scale=1.35,
                        guidance_decay="interval",
                        guidance_scale_pow=0.0,
                        randomize_temperature=1.0,
                        softmax_temperature_annealing=True,
                        num_sample_steps=num_sample_steps,
                        device=device
                    )
                    Image.fromarray(generated_image[0]).save(os.path.join(save_dir, f"{'diffusion' if is_diffusion else 'maskgit'}_{solver}_{'noregret' if noregret else 'regret'}_{get_name}_{num_sample_steps}.png"))
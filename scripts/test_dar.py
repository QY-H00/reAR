import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import demo_util
from demo_util import get_tokenizer
import os
from data.imagenet_classes import imagenet_idx2classname
import time

if __name__ == "__main__":
    guidance_scale = 3.0
    iters = 800000
    T = 1.0
    num_sample_steps = 16
    guidance_scale_pow = 0.0
    kv_cache = True
    guidance_decay = "linear"

    # download the maskgit-vq tokenizer
    # hf_hub_download(repo_id="fun-research/TiTok", filename=f"maskgit-vqgan-imagenet-f16-256.bin", local_dir="./")

    # load config
    config = demo_util.get_config("configs/training/generator/dar_titok_l32.yaml")
    config.experiment.generator_checkpoint = "/home/ubuntu/dAR/temp/dar_titok_l32_main_more_warmup/checkpoint-250000/model.safetensors"
    config.model.generator.hidden_size = 1024
    config.model.generator.num_hidden_layers = 24
    config.model.generator.num_attention_heads = 16
    config.model.generator.intermediate_size = 4096

    device = "cuda"
    # maskgit-vq as tokenizer
    tokenizer = get_tokenizer(config)
    generator = demo_util.get_dar_generator(config)
    tokenizer.to(device)
    generator.to(device)

    # generate an image
    torch.manual_seed(42)
    # sample_labels = torch.randint(0, 1000, size=(6,)).to(device)
    sample_labels = torch.tensor([1, 7, 282, 604, 724, 179, 751, 404, 850, torch.randint(0, 999, size=(1,))]).to(device)
    start_time = time.time()
    generated_image = demo_util.sample_fn(
        generator=generator,
        tokenizer=tokenizer,
        # labels=sample_labels,
        randomize_temperature=T,
        guidance_scale=guidance_scale,
        guidance_scale_pow=guidance_scale_pow,
        guidance_decay=guidance_decay,
        device=device,
        num_sample_steps=num_sample_steps,
        kv_cache=kv_cache
    )
    end_time = time.time()
    print(f"Time taken for generation: {end_time - start_time} seconds")

    # create save dir
    os.makedirs("temp/vis", exist_ok=True)
    # Create a grid of images with labels
    grid_size = int(np.ceil(np.sqrt(len(sample_labels))))
    grid_width = grid_size * generated_image[0].shape[1]
    grid_height = grid_size * (generated_image[0].shape[0] + 30)  # Extra space for text
    
    # Create a blank canvas
    grid_image = Image.new('RGB', (grid_width, grid_height), color=(255, 255, 255))
    
    for i, label in enumerate(sample_labels):
        # Convert numpy array to PIL Image
        img = Image.fromarray(generated_image[i])
        
        # Calculate position in grid
        row = i // grid_size
        col = i % grid_size
        x = col * img.width
        y = row * (img.height + 30)  # Extra space for text
        
        # Create a temporary image with space for text
        temp_img = Image.new('RGB', (img.width, img.height + 30), color=(255, 255, 255))
        temp_img.paste(img, (0, 30))
        
        # Add text with class name
        draw = ImageDraw.Draw(temp_img)
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except IOError:
            font = ImageFont.load_default()
        
        class_name = imagenet_idx2classname[label.item()][:20]
        draw.text((10, 5), str(label.item()) + " " + class_name, fill=(0, 0, 0), font=font)
        
        # Paste onto the grid
        grid_image.paste(temp_img, (x, y))
    
    # Save the grid image
    grid_image.save(f"temp/vis/steps{num_sample_steps}_scale{guidance_scale}_T{T}_iters{iters}{'_kv_cache' if kv_cache else ''}_{guidance_decay}.png")
import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import demo_util
from demo_util import get_tokenizer
import os
from data.imagenet_classes import imagenet_idx2classname
import time

if __name__ == "__main__":
    guidance_scale = 16.25
    iters = 250000
    T = 1.0
    num_sample_steps = 256
    guidance_scale_pow = 2.75
    kv_cache = True
    guidance_decay = "power-cosine"
    use_annealed_temp = True
    fix_orders = False
    rope_type = "2d"
    head_type="distributed"
    maskgit_sampling = False
    top_k = None
    top_p = 0.95
    inference_k_tokens = 32

    # download the maskgit-vq tokenizer
    # hf_hub_download(repo_id="fun-research/TiTok", filename=f"maskgit-vqgan-imagenet-f16-256.bin", local_dir="./")

    # load config
    config = demo_util.get_config("configs/training/generator/dar_maskgitvq.yaml")
    config.experiment.generator_checkpoint = "/home/ubuntu/dAR/temp/dar_maskgitvq_ema_rope_2d_diffusion_head_tuned2/checkpoint-100000/ema_model/pytorch_model.bin"
    config.model.generator.hidden_size = 768
    config.model.generator.num_hidden_layers = 24
    config.model.generator.num_attention_heads = 16
    config.model.generator.intermediate_size = 3072
    config.model.generator.rope_type = rope_type
    config.model.generator.head_type = head_type

    device = "cuda"
    # maskgit-vq as tokenizer
    tokenizer = get_tokenizer(config)
    generator = demo_util.get_dar_generator(config)
    tokenizer.to(device)
    generator.to(device)

    # Define seeds and classes to test
    seeds = list(range(42, 44))  # Seeds 42 to 49 (8 seeds)
    sample_classes = [1, 7, 282, 604, 724, 179, 751, 404, 850, 123]  # 10 classes
    
    print(f"Generating images for {len(sample_classes)} classes across {len(seeds)} seeds")
    print(f"Seeds: {seeds}")
    print(f"Classes: {sample_classes}")
    
    # Generate images for all combinations
    all_generated_images = [[] for _ in range(len(sample_classes))]
    class_names = []
    
    total_start_time = time.time()
    
    # Get class names first
    for class_idx, class_label in enumerate(sample_classes):
        class_name = imagenet_idx2classname[class_label]
        class_names.append(f"{class_label}: {class_name[:20]}")
    
    # Generate images for all classes in batch for each seed
    for seed_idx, seed in enumerate(seeds):
        print(f"\nGenerating for seed {seed} ({seed_idx+1}/{len(seeds)})")
        
        # Set seed for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Create tensor with all class labels
        sample_labels = torch.tensor(sample_classes).to(device)
        
        start_time = time.time()
        generated_images = demo_util.sample_fn(
            generator=generator,
            tokenizer=tokenizer,
            labels=sample_labels,
            randomize_temperature=T,
            guidance_scale=guidance_scale,
            guidance_scale_pow=guidance_scale_pow,
            guidance_decay=guidance_decay,
            device=device,
            num_sample_steps=num_sample_steps,
            kv_cache=kv_cache,
            fix_orders=fix_orders,
            use_annealed_temp=use_annealed_temp,
            maskgit_sampling=maskgit_sampling,
            top_k=top_k,
            top_p=top_p,
            inference_k_tokens=inference_k_tokens
        )
        end_time = time.time()
        print(f"  Time: {end_time - start_time:.2f}s")
        
        # Store the generated images
        for class_idx, img in enumerate(generated_images):
            all_generated_images[class_idx].append(img)
    
    total_end_time = time.time()
    print(f"\nTotal generation time: {total_end_time - total_start_time:.2f} seconds")
    
    # Create save dir
    os.makedirs("temp/vis_temp_maskgitvq", exist_ok=True)
    
    # Create grid layout: rows = classes, columns = seeds
    num_classes = len(sample_classes)
    num_seeds = len(seeds)
    
    # Get image dimensions (assuming all images are the same size)
    img_height, img_width = all_generated_images[0][0].shape[:2]
    
    # Calculate grid dimensions
    label_height = 40  # Height for class labels
    seed_label_height = 25  # Height for seed labels
    grid_width = num_seeds * img_width
    grid_height = num_classes * img_height + label_height + seed_label_height
    
    # Create a blank canvas
    grid_image = Image.new('RGB', (grid_width, grid_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(grid_image)
    
    # Try to load a font
    try:
        font = ImageFont.truetype("arial.ttf", 16)
        small_font = ImageFont.truetype("arial.ttf", 12)
    except IOError:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()
    
    # Add seed labels at the top
    for seed_idx, seed in enumerate(seeds):
        x = seed_idx * img_width + img_width // 2
        draw.text((x - 20, 5), f"Seed {seed}", fill=(0, 0, 0), font=small_font)
    
    # Add images to grid
    for class_idx, (class_images, class_name) in enumerate(zip(all_generated_images, class_names)):
        # Calculate y position for this class row
        y_offset = seed_label_height + class_idx * img_height
        
        # Add class label on the left
        draw.text((5, y_offset + img_height // 2), class_name, fill=(0, 0, 0), font=font)
        
        for seed_idx, img_array in enumerate(class_images):
            # Convert numpy array to PIL Image
            img = Image.fromarray(img_array)
            
            # Calculate position in grid
            x = seed_idx * img_width
            y = y_offset
            
            # Paste image onto the grid
            grid_image.paste(img, (x, y))
    
    # Save the grid image
    filename = f"temp/vis_temp_maskgitvq/multi_seed_grid_steps{num_sample_steps}_scale{guidance_scale}_T{T}_iters{iters}{'_kv_cache' if kv_cache else ''}_{guidance_decay}{'_fix_orders' if fix_orders else ''}_{rope_type}{'_annealed_temp' if use_annealed_temp else ''}{'_maskgit_sampling' if maskgit_sampling else ''}{'_top_k' if top_k is not None else ''}{'_top_p' if top_p is not None else ''}.png"
    grid_image.save(filename)
    
    print(f"\nGrid image saved as: {filename}")
    print(f"Grid layout: {num_classes} classes × {num_seeds} seeds")
    print("Each row represents a class, each column represents a different seed")
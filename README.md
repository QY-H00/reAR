# reAR: Rethink Visual Autoregressive Generation via Robust Embedding Regularization

This repository contains the official implementation of the paper **"reAR: Rethink Standard Visual Autoregressive Generation via Robust Embedding Regularization"**.

## 🎯 Overview

We rethink the challenges in visual autoregressive generation and propose a simple yet effective technique that enables standard visual AR (rasterization order) to outperform methods with advanced tokenizers and generation orders. Our approach introduces robust embedding regularization that significantly improves the quality and efficiency of visual autoregressive models.


## 🚀 Quick Start

### 1. Environment Setup

```bash
# Create conda environment
conda create -n rear python=3.10
conda activate rear

# Install dependencies
pip install -r requirements.txt
```

### 2. Project Structure

Set up your project directory as follows:

```
re-ar/
├── configs/                    # Configuration files
├── dataset/                    # Dataset directory
│   ├── pretokenized/          # Pre-tokenized data (recommended)
│   │   └── maskgitvq.jsonl   # Download from HuggingFace
│   └── imagenet_shard/        # Original ImageNet (optional)
│       ├── train/
│       │   ├── imagenet-train-0000.tar
│       │   ├── imagenet-train-0001.tar
│       │   └── ...
│       └── val/
│           ├── imagenet-val-0000.tar
│           ├── imagenet-val-0001.tar
│           └── ...
├── ckpt/                      # Model checkpoints
│   └── maskgitvq.bin         # Download from HuggingFace
└── scripts_bash/             # Training and evaluation scripts
```

### 3. Data Preparation

#### Option A: Pre-tokenized Data (Recommended)
Download the pre-tokenized files for faster training:
```bash
# Download pre-tokenized data
wget https://huggingface.co/yucornetto/RAR/resolve/main/maskgitvq.jsonl -O dataset/pretokenized/maskgitvq.jsonl
```

#### Option B: Original ImageNet
For original ImageNet in webdataset format, follow the [TiTok instructions](https://github.com/bytedance/1d-tokenizer/blob/main/README_TiTok.md).

### 4. Model Checkpoints

Download the MaskGiT-VQGAN tokenizer:
```bash
# Download tokenizer checkpoint
wget https://huggingface.co/fun-research/TiTok/resolve/main/maskgit-vqgan-imagenet-f16-256.bin -O ckpt/maskgitvq.bin
```

## 🏋️ Training

### Single Command Training

```bash
bash scripts_bash/rear_train.sh
```

### Manual Training Setup

If you prefer to run training manually, here's the complete setup:

```bash
# Navigate to project directory
cd ~/re-ar

# Activate environment
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate rear

# Set environment variables
export PYTHONPATH=$(pwd)
export WANDB_INIT_TIMEOUT=300

# Configure training parameters
entity="your_wandb_entity"  # Replace with your Weights & Biases entity
config_name='rear_l'

# Launch training
accelerate launch \
    --num_machines=1 --num_processes=8 --machine_rank=0 \
    --main_process_ip=127.0.0.1 --main_process_port=9999 --same_network \
    scripts/train_rear.py config="configs/${config_name}.yaml" \
    experiment.entity="${entity}" \
    experiment.output_dir="temp/${config_name}"
```

## 📊 Evaluation

### Setup ADM Evaluation

```bash
# Clone ADM evaluation repository
git clone https://github.com/openai/guided-diffusion.git

# Download reference batches
wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz
```

### Run Evaluation

```bash
# Quick evaluation
bash scripts_bash/rear_test.sh
```

### Manual Evaluation

```bash
# Navigate to project directory
cd ~/re-ar

# Activate environment
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate rear

# Set environment variables
export PYTHONPATH=$(pwd)
export WANDB_INIT_TIMEOUT=300

# Configure evaluation parameters
config_name="rear_l"  # Change to your config
output_dir="${config_name}"
checkpoint_path="path/to/your/checkpoint"  # Specify your checkpoint path

# Generate samples
torchrun --nnodes=1 --nproc_per_node=8 --rdzv-endpoint=localhost:19999 \
    scripts/sample_imagenet_rear.py config="configs/training/generator/${config_name}.yaml" \
    experiment.output_dir="${output_dir}" \
    experiment.generator_checkpoint="${checkpoint_path}"

# Evaluate samples
python3 guided-diffusion/evaluations/evaluator.py \
    VIRTUAL_imagenet256_labeled.npz ${output_dir}.npz
```

## 📈 Results

Our method achieves state-of-the-art performance on ImageNet generation benchmarks. For detailed results and comparisons, please refer to our paper.

## 🤝 Contributing

We welcome contributions! Please feel free to submit issues and pull requests.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

We thank the following projects for their excellent work:
- [RAR](https://github.com/bytedance/1d-tokenizer/blob/main/README_RAR.md): Randomized Autoregressive Visual Generation
- [TiTok](https://github.com/bytedance/1d-tokenizer/blob/main/README_TiTok.md): An Image is Worth 32 Tokens for Reconstruction and Generation
- [MaskBit](https://github.com/markweberdev/maskbit/tree/main): Embedding-free Image Generation via Bit Tokens
- [REPA](https://github.com/sihyun-yu/REPA/tree/main): Representation Alignment for Generation:
Training Diffusion Transformers Is Easier Than You Think

**Note**: This is a placeholder citation. Please update with the actual paper details when available.
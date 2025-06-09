#PBS -N zexp_dgen_test
#PBS -S /bin/bash
#PBS -l select=1:ncpus=6:mem=45gb:ngpus=1:host=cvml11

tiny=True

nvidia-smi
cd ~/dGen
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate titok

export PYTHONPATH=$(pwd)
export WANDB_INIT_TIMEOUT=300

torchrun --nnodes=1 --nproc_per_node=2 --rdzv-endpoint=localhost:9999 sample_imagenet_dar.py config=configs/training/generator/dgen.yaml \
    experiment.output_dir="dgen_b" \
    experiment.generator_checkpoint="/home/qiyuan/dGen/temp/test_add_parameterization_again/checkpoint-150000/unwrapped_model/pytorch_model.bin" \
    model.generator.hidden_size=768 \
    model.generator.num_hidden_layers=24 \
    model.generator.num_attention_heads=16 \
    model.generator.intermediate_size=3072 \
    model.generator.randomize_temperature=1.0 \
    model.generator.guidance_scale=16.0 \
    model.generator.guidance_scale_pow=2.75 \
    model.generator.num_sample_steps=8 \
    model.generator.tiny=$tiny \

python3 guided-diffusion/evaluations/evaluator.py VIRTUAL_imagenet256_labeled.npz dgen_b.npz
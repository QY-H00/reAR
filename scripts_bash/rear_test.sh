cd ~/re-ar
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate titok

export PYTHONPATH=$(pwd)
export WANDB_INIT_TIMEOUT=300

config_name="rear_l"
output_dir="${config_name}"
checkpoint_path="ckpt/${config_name}.safetensors"

torchrun --nnodes=1 --nproc_per_node=8 --rdzv-endpoint=localhost:19999 scripts/sample_imagenet_rear.py config=configs/training/generator/${config_name}.yaml \
    experiment.output_dir="${output_dir}" \
    experiment.generator_checkpoint="${checkpoint_path}"

python3 guided-diffusion/evaluations/evaluator.py VIRTUAL_imagenet256_labeled.npz ${output_dir}.npz
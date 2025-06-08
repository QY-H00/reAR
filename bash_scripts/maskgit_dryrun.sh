#PBS -N zexp_maskgit_scratch
#PBS -S /bin/bash
#PBS -l select=1:ncpus=48:mem=360gb:ngpus=4:host=cvml04

config_name='maskgit'
tag="scratch_titok"

nvidia-smi
cd ~/dGen
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate titok

export PYTHONPATH=$(pwd)
export WANDB_INIT_TIMEOUT=300

WANDB_MODE=offline CUDA_VISIBLE_DEVICES=1 accelerate launch \
    --num_machines=1 --num_processes=1 --machine_rank=0 \
    --main_process_ip=127.0.0.1 --main_process_port=9999 --same_network \
    scripts/train_maskgit.py config="configs/training/generator/${config_name}.yaml" \
    experiment.project="dgen_test" \
    experiment.name="${config_name}_${tag}" \
    experiment.entity="hodavid538" \
    experiment.output_dir="temp/${config_name}_${tag}" \
    \
    experiment.save_every=5 \
    experiment.eval_every=5 \
    experiment.generate_every=2 \
    experiment.log_every=1 \
    \
    model.generator.hidden_size=768 \
    model.generator.num_hidden_layers=24 \
    model.generator.num_attention_heads=16 \
    model.generator.intermediate_size=3072 \
    training.per_gpu_batch_size=8 \
    training.gradient_accumulation_steps=1 \
    training.max_train_steps=800_000 \
    dataset.params.train_shards_path_or_url="/mnt/rdata8/imagenet_wds/imagenet-train-{000000..000320}.tar" \
    dataset.params.eval_shards_path_or_url="/mnt/rdata8/imagenet_wds/imagenet-val-{000000..000049}.tar" \
    
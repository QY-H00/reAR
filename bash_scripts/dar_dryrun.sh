#PBS -N zexp_ar_test
#PBS -S /bin/bash
#PBS -l select=1:ncpus=48:mem=360gb:ngpus=8:host=cvml10

config_name='dar'
tag="dar_test_bz8_ga4"

nvidia-smi
cd ~/dGen
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate titok

export PYTHONPATH=$(pwd)
export WANDB_INIT_TIMEOUT=300

WANDB_MODE=offline CUDA_VISIBLE_DEVICES=2,3 accelerate launch \
    --num_machines=1 --num_processes=2 --machine_rank=0 \
    --main_process_ip=127.0.0.1 --main_process_port=9999 --same_network \
    scripts/train_dar.py config="configs/training/generator/${config_name}.yaml" \
    experiment.project="dgen_test" \
    experiment.name="${config_name}_${tag}" \
    experiment.entity="hodavid538" \
    experiment.output_dir="temp/${config_name}_${tag}" \
    model.generator.hidden_size=768 \
    model.generator.num_hidden_layers=24 \
    model.generator.num_attention_heads=16 \
    model.generator.intermediate_size=3072 \
    training.per_gpu_batch_size=64 \
    training.gradient_accumulation_steps=4 \
    lr_scheduler.params.learning_rate=1e-4 \
    lr_scheduler.params.warmup_steps=200_000 \
    training.max_train_steps=1_000_000 \
    dataset.params.train_shards_path_or_url="/mnt/rdata8/imagenet_wds/imagenet-train-{000000..000320}.tar" \
    dataset.params.eval_shards_path_or_url="/mnt/rdata8/imagenet_wds/imagenet-val-{000000..000049}.tar" \
    \
    experiment.save_every=1000 \
    experiment.eval_every=1000 \
    experiment.generate_every=1 \
    experiment.log_every=100 \
    \
#PBS -N zexp_ar_test
#PBS -S /bin/bash
#PBS -l select=1:ncpus=48:mem=360gb:ngpus=8:host=cvml10

config_name='dar_maskgitvq'
tag="dryrun_test"

nvidia-smi
cd ~/dAR
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate dar

export PYTHONPATH=$(pwd)
export WANDB_INIT_TIMEOUT=300

export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,mlx5_4:1,mlx5_5:1 
export NCCL_IB_DISABLE=0 
export NCCL_SOCKET_IFNAME=bond0 
export NCCL_DEBUG=INFO

# accelerate launch \
accelerate launch \
    --num_machines=1 --num_processes=8 --machine_rank=0 \
    --main_process_ip=127.0.0.1 --main_process_port=9999 --same_network \
    scripts/train_dar.py config="configs/training/generator/${config_name}.yaml" \
    experiment.project="dar" \
    experiment.name="${config_name}_${tag}" \
    experiment.entity="hodavid538" \
    experiment.output_dir="temp/${config_name}_${tag}" \
    training.enable_swanlab=True \
    model.generator.hidden_size=768 \
    model.generator.num_hidden_layers=19 \
    model.generator.num_attention_heads=16 \
    model.generator.intermediate_size=3072 \
    model.generator.rope_type="2d" \
    model.generator.head_type="distributed" \
    model.generator.k_tokens=4 \
    training.per_gpu_batch_size=256 \
    training.gradient_accumulation_steps=1 \
    lr_scheduler.params.learning_rate=1e-4 \
    lr_scheduler.params.warmup_steps=62_500 \
    training.max_train_steps=250_000 \
    training.use_ema=True \
    dataset.params.train_shards_path_or_url="/mnt/rdata8/imagenet_wds/imagenet-train-{000000..000320}.tar" \
    dataset.params.eval_shards_path_or_url="/mnt/rdata8/imagenet_wds/imagenet-val-{000000..000049}.tar" \
    \
    experiment.save_every=1000 \
    experiment.eval_every=1000 \
    experiment.generate_every=100 \
    experiment.log_every=100 \
    \
    losses.no_mask=False \
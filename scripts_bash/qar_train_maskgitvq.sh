#PBS -N zexp_ar_test
#PBS -S /bin/bash
#PBS -l select=1:ncpus=48:mem=360gb:ngpus=8:host=cvml10

config_name='qar_maskgitvq'
tag="v1_02"
fix_orders=False
warmup_steps=10_000
max_train_steps=250_000
lr=2e-4
end_lr=1e-5

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

export TORCHDYNAMO_VERBOSE=1

# accelerate launch \
accelerate launch \
    --num_machines=1 --num_processes=8 --machine_rank=0 \
    --main_process_ip=127.0.0.1 --main_process_port=9999 --same_network \
    scripts/train_qar.py config="configs/training/generator/${config_name}.yaml" \
    experiment.project="qar" \
    experiment.name="${config_name}_${tag}" \
    experiment.entity="hodavid538" \
    experiment.output_dir="temp/${config_name}_${tag}" \
    training.enable_swanlab=True \
    model.generator.hidden_size=768 \
    model.generator.num_hidden_layers=20 \
    model.generator.num_attention_heads=16 \
    model.generator.intermediate_size=3072 \
    model.generator.fix_orders=${fix_orders} \
    training.per_gpu_batch_size=256 \
    training.gradient_accumulation_steps=1 \
    optimizer.params.learning_rate=${lr} \
    lr_scheduler.params.learning_rate=${lr} \
    lr_scheduler.params.warmup_steps=${warmup_steps} \
    lr_scheduler.params.end_lr=${end_lr} \
    training.max_train_steps=${max_train_steps} \
    training.use_ema=False \
    dataset.params.train_shards_path_or_url="/mnt/rdata8/imagenet_wds/imagenet-train-{000000..000320}.tar" \
    dataset.params.eval_shards_path_or_url="/mnt/rdata8/imagenet_wds/imagenet-val-{000000..000049}.tar" \
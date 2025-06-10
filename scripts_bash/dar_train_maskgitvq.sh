config_name='dar_maskgitvq'
tag="main"

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

learning_rate=4e-4
end_lr=1e-5
max_train_steps=250_000
warmup_steps=62_500

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
    model.generator.num_hidden_layers=24 \
    model.generator.num_attention_heads=16 \
    model.generator.intermediate_size=3072 \
    training.per_gpu_batch_size=256 \
    training.gradient_accumulation_steps=1 \
    optimizer.params.learning_rate=${learning_rate} \
    lr_scheduler.params.learning_rate=${learning_rate} \
    lr_scheduler.params.warmup_steps=${warmup_steps} \
    lr_scheduler.params.end_lr=${end_lr} \
    training.max_train_steps=${max_train_steps} \
    dataset.params.train_shards_path_or_url="/home/ubuntu/dataset/imagenet1k_wds/imagenet1k-train-{000000..001023}.tar" \
    dataset.params.eval_shards_path_or_url="/home/ubuntu/dataset/imagenet1k_wds/imagenet1k-validation-{000000..000063}.tar" \
    
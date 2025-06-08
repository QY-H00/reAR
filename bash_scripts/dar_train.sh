#PBS -N dar_b_full
#PBS -S /bin/bash
#PBS -l select=1:ncpus=48:mem=360gb:ngpus=8:host=cvml06

config_name='dar'
tag="b_titokl32_agressive_lr4e-4_600k"

nvidia-smi
cd ~/dGen
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate titok

export PYTHONPATH=$(pwd)
export WANDB_INIT_TIMEOUT=300
export WANDB_API_TIMEOUT=300

learning_rate=4e-4
end_lr=1e-6
max_train_steps=600_000
warmup_steps=150_000

accelerate launch \
    --num_machines=1 --num_processes=8 --machine_rank=0 \
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
    training.gradient_accumulation_steps=1 \
    optimizer.params.learning_rate=${learning_rate} \
    lr_scheduler.params.learning_rate=${learning_rate} \
    lr_scheduler.params.warmup_steps=${warmup_steps} \
    lr_scheduler.params.end_lr=${end_lr} \
    training.max_train_steps=${max_train_steps} \
    dataset.params.train_shards_path_or_url="/mnt/rdata8/imagenet_wds/imagenet-train-{000000..000320}.tar" \
    dataset.params.eval_shards_path_or_url="/mnt/rdata8/imagenet_wds/imagenet-val-{000000..000049}.tar" \
    
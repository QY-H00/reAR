#PBS -N zexp_dgen_postrain
#PBS -S /bin/bash
#PBS -l select=1:ncpus=48:mem=360gb:ngpus=8:host=cvml03

config_name='maskgit'
tag="scratch_titok_diffusion_postrain"

nvidia-smi
cd ~/dGen
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate titok

export PYTHONPATH=$(pwd)
export WANDB_INIT_TIMEOUT=300

accelerate launch \
    --num_machines=1 --num_processes=4 --machine_rank=0 \
    --main_process_ip=127.0.0.1 --main_process_port=9999 --same_network \
    scripts/train_maskgit.py config="configs/training/generator/${config_name}.yaml" \
    experiment.project="dgen_test" \
    experiment.name="${config_name}_${tag}" \
    experiment.entity="hodavid538" \
    experiment.output_dir="temp/${config_name}_${tag}" \
    experiment.init_weight="/home/qiyuan/dGen/pretrained_checkpoints/generator_titok_b64_imagenet/model.safetensors" \
    \
    losses.is_diffusion=True \
    \
    model.generator.hidden_size=768 \
    model.generator.num_hidden_layers=24 \
    model.generator.num_attention_heads=16 \
    model.generator.intermediate_size=3072 \
    training.per_gpu_batch_size=64 \
    training.gradient_accumulation_steps=1 \
    training.max_train_steps=800_000 \
    dataset.params.train_shards_path_or_url="/mnt/rdata8/imagenet_wds/imagenet-train-{000000..000320}.tar" \
    dataset.params.eval_shards_path_or_url="/mnt/rdata8/imagenet_wds/imagenet-val-{000000..000049}.tar" \
    
#PBS -N zexp_dgen_scratch
#PBS -S /bin/bash
#PBS -l select=1:ncpus=48:mem=360gb:ngpus=8:host=cvml06

config_name='maskgit'
tag="scratch_titokb64_dd_128bsz_l_nonuniform_loss"

nvidia-smi
cd ~/dGen
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate titok

export PYTHONPATH=$(pwd)
export WANDB_INIT_TIMEOUT=300

is_diffusion=True
no_regret=False
uniform_loss_weights=False

accelerate launch \
    --num_machines=1 --num_processes=8 --machine_rank=0 \
    --main_process_ip=127.0.0.1 --main_process_port=9999 --same_network \
    scripts/train_maskgit.py config="configs/training/generator/${config_name}.yaml" \
    experiment.project="dgen_test" \
    experiment.name="${config_name}_${tag}" \
    experiment.entity="hodavid538" \
    experiment.output_dir="temp/${config_name}_${tag}" \
    \
    losses.is_diffusion=${is_diffusion} \
    losses.uniform_loss_weights=${uniform_loss_weights} \
    \
    model.generator.is_diffusion=${is_diffusion} \
    model.generator.noregret=${no_regret} \
    model.generator.guidance_decay="interval" \
    model.generator.guidance_scale=1.5 \
    model.generator.randomize_temperature=1.0 \
    model.generator.softmax_temperature_annealing=True \
    \
    model.generator.hidden_size=1024 \
    model.generator.num_hidden_layers=24 \
    model.generator.num_attention_heads=16 \
    model.generator.intermediate_size=4096 \
    training.per_gpu_batch_size=128 \
    training.gradient_accumulation_steps=1 \
    lr_scheduler.params.learning_rate=1e-4 \
    lr_scheduler.params.warmup_steps=200_000 \
    training.max_train_steps=1_000_000 \
    dataset.params.train_shards_path_or_url="/mnt/rdata8/imagenet_wds/imagenet-train-{000000..000320}.tar" \
    dataset.params.eval_shards_path_or_url="/mnt/rdata8/imagenet_wds/imagenet-val-{000000..000049}.tar" \
    
config_name='par_maskgitvq'
tag="en0_de18_0-75_uniform_sizeL_try"

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

layers=24
decode_layer=18
hidden_size=1024
intermediate_size=4096
accumulation_steps=2
per_gpu_batch_size=128

accelerate launch \
    --num_machines=1 --num_processes=8 --machine_rank=0 \
    --main_process_ip=127.0.0.1 --main_process_port=9999 --same_network \
    scripts/train_rar.py config="configs/training/generator/${config_name}.yaml" \
    experiment.project="par" \
    experiment.name="${config_name}_${tag}" \
    experiment.entity="hodavid538" \
    experiment.output_dir="temp/${config_name}_${tag}" \
    training.enable_swanlab=True \
    model.generator.hidden_size=${hidden_size} \
    model.generator.num_hidden_layers=${layers} \
    model.generator.num_attention_heads=16 \
    model.generator.intermediate_size=${intermediate_size} \
    model.generator.perturb_mode="switch" \
    model.generator.randomness_anneal_start=0 \
    model.generator.randomness_anneal_end=187500 \
    model.generator.transition_alignment=True \
    model.generator.decode_align_layer_idx=${decode_layer} \
    model.generator.transition_alignment_weight=1.0 \
    training.per_gpu_batch_size=${per_gpu_batch_size} \
    training.gradient_accumulation_steps=${accumulation_steps}
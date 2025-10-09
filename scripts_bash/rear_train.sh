cd ~/re-ar
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate rear

export PYTHONPATH=$(pwd)
export WANDB_INIT_TIMEOUT=300

entity= # TYPE YOUR ENTITY NAME HERE
config_name='rear_l'

accelerate launch \
    --num_machines=1 --num_processes=8 --machine_rank=0 \
    --main_process_ip=127.0.0.1 --main_process_port=9999 --same_network \
    scripts/train_rear.py config="configs/${config_name}.yaml" \
    experiment.entity="${entity}" \
    experiment.output_dir="temp/${config_name}" \
    training.enable_swanlab=True \
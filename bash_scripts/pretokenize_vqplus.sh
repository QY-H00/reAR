#PBS -N zexp_maskgit_scratch
#PBS -S /bin/bash
#PBS -l select=1:ncpus=24:mem=180gb:ngpus=4:host=cvml10


PATH_TO_IMAGENET="/mnt/rdata8/imagenet"
PATH_TO_SAVE_JSONL="/mnt/rdata8/imagenet_pretokenized/vqplus"

nvidia-smi
cd ~/dGen
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate titok

export PYTHONPATH=$(pwd)

# PATH_TO_IMAGENET="/mnt/rdata8/imagenet"
# PATH_TO_SAVE_JSONL="/home/qiyuan/dGen/temp/test_pretokenize"

torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 --rdzv-endpoint=localhost:9999 \
    scripts/pretokenization.py \
    --config configs/training/generator/dar_vqplus.yaml \
    --img_size 256 \
    --batch_size 8 \
    --ten_crop \
    --data_path ${PATH_TO_IMAGENET} --cached_path ${PATH_TO_SAVE_JSONL}
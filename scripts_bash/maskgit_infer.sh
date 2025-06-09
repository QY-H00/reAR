nvidia-smi
cd ~/dGen
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate titok

export PYTHONPATH=$(pwd)

python scripts/test_maskgit.py
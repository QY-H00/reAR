nvidia-smi
cd ~/dAR
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate dar

is_diffusion=True
if [ "$is_diffusion" = "True" ]; then
    prefix="ours"
else
    prefix="baseline"
fi

export PYTHONPATH=$(pwd)
export WANDB_INIT_TIMEOUT=300

export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,mlx5_4:1,mlx5_5:1 
export NCCL_IB_DISABLE=0 
export NCCL_SOCKET_IFNAME=bond0 
export NCCL_DEBUG=INFO

# Define parameter arrays for sweeping
temperatures=(1.0)
steps=(32)
scales=(3.0)
guidance_decay="linear"
guidance_scale_pow=2.75
fix_orders=False
use_annealed_temps=(False)
ckpt_name="dar_titok_l32_ema_rope1d_ema_250000.bin"
rope_type="1d"
maskgit_sampling=False
top_k=400
top_p=None

# Initialize port counter for unique localhost endpoints
port=19900

echo " Running on ${ckpt_name}"

# Loop through all parameter combinations
for randomize_temperature in "${temperatures[@]}"; do
    for num_sample_steps in "${steps[@]}"; do
        for guidance_scale in "${scales[@]}"; do
            for use_annealed_temp in "${use_annealed_temps[@]}"; do

                output_dir="${prefix}_steps${num_sample_steps}_scale${guidance_scale}_temperature${randomize_temperature}_annealed${use_annealed_temp}"
                
                echo "Running with parameters:"
                echo "  Temperature: ${randomize_temperature}"
                echo "  Steps: ${num_sample_steps}"
                echo "  Decay: ${guidance_decay}"
                echo "  Guidance Scale: ${guidance_scale}"
                echo "  Output Directory: ${output_dir}"
                echo "  Port: ${port}"
                echo "  Maskgit Sampling: ${maskgit_sampling}"
                echo "  Top-k: ${top_k}"
                echo "  Top-p: ${top_p}"
                
                # Run the sampling with unique localhost port
                torchrun --nnodes=1 --nproc_per_node=8 --rdzv-endpoint=localhost:${port} scripts/sample_imagenet_dar.py config=configs/training/generator/dar_titok_l32.yaml \
                    experiment.output_dir="${output_dir}" \
                    experiment.tokenizer_checkpoint=yucornetto/tokenizer_titok_l32_imagenet \
                    experiment.generator_checkpoint=/home/qiyuan/dGen/temp/dAR_experiments/${ckpt_name} \
                    model.generator.hidden_size=1024 \
                    model.generator.num_hidden_layers=24 \
                    model.generator.num_attention_heads=16 \
                    model.generator.intermediate_size=4096 \
                    model.generator.randomize_temperature=${randomize_temperature} \
                    model.generator.guidance_decay=${guidance_decay} \
                    model.generator.guidance_scale=${guidance_scale} \
                    model.generator.guidance_scale_pow=${guidance_scale_pow} \
                    model.generator.num_steps=${num_sample_steps} \
                    model.generator.fix_orders=${fix_orders} \
                    model.generator.use_annealed_temp=${use_annealed_temp} \
                    model.generator.rope_type=${rope_type} \
                    model.generator.maskgit_sampling=${maskgit_sampling} \
                    model.generator.top_k=${top_k} \
                    model.generator.top_p=${top_p}

                # Run evaluation
                python3 guided-diffusion/evaluations/evaluator.py VIRTUAL_imagenet256_labeled.npz ${output_dir}.npz

                # Clean up
                rm -rf ${output_dir}
                rm ${output_dir}.npz
                
                # Increment port for next run
                port=$((port + 1))
                
                echo "Finished run with parameters: temp=${randomize_temperature}, steps=${num_sample_steps}, scale=${guidance_scale}"
                echo "---------------------------------------------------"
            done
        done
    done
done
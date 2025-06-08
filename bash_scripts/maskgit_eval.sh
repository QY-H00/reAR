#!/bin/bash
#PBS -N zexp_dgen_test
#PBS -S /bin/bash
#PBS -l select=1:ncpus=48:mem=360gb:ngpus=8:host=cvml06

nvidia-smi
cd ~/dGen
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate titok

is_diffusion=True
if [ "$is_diffusion" = "True" ]; then
    prefix="ours"
else
    prefix="baseline"
fi

export PYTHONPATH=$(pwd)
export WANDB_INIT_TIMEOUT=300

# Define parameter arrays for sweeping
temperatures=(1.0)
steps=(16)
scales=(0.0)
regret_options=(True)
guidance_decay="interval"
guidance_scale_pow=0.0
softmax_temperature_annealing=False

# Initialize port counter for unique localhost endpoints
port=19900

# Loop through all parameter combinations
for randomize_temperature in "${temperatures[@]}"; do
    for num_sample_steps in "${steps[@]}"; do
        for guidance_scale in "${scales[@]}"; do
            for no_regret in "${regret_options[@]}"; do
                # Set noregret string for output directory naming
                if [ "$no_regret" = "True" ]; then
                    noregret_str="noregret"
                else
                    noregret_str=""
                fi

                output_dir="${prefix}_steps${num_sample_steps}_scale${guidance_scale}_temperature${randomize_temperature}_${noregret_str}"
                
                echo "Running with parameters:"
                echo "  Temperature: ${randomize_temperature}"
                echo "  Steps: ${num_sample_steps}"
                echo "  Guidance Scale: ${guidance_scale}"
                echo "  No Regret: ${no_regret}"
                echo "  Output Directory: ${output_dir}"
                echo "  Port: ${port}"
                
                # Run the sampling with unique localhost port
                torchrun --nnodes=1 --nproc_per_node=2 --rdzv-endpoint=localhost:${port} sample_imagenet_maskgit.py config=configs/training/generator/maskgit.yaml \
                    experiment.output_dir="${output_dir}" \
                    experiment.tokenizer_checkpoint=yucornetto/tokenizer_titok_b64_imagenet \
                    experiment.generator_checkpoint=temp/maskgit_scratch_titokb64_dd_128bsz_l_nonuniform_loss/checkpoint-400000/ema_model/pytorch_model.bin \
                    model.generator.hidden_size=1024 \
                    model.generator.num_hidden_layers=24 \
                    model.generator.num_attention_heads=16 \
                    model.generator.intermediate_size=4096 \
                    model.generator.randomize_temperature=${randomize_temperature} \
                    model.generator.softmax_temperature_annealing=${softmax_temperature_annealing} \
                    model.generator.is_diffusion=${is_diffusion} \
                    model.generator.noregret=${no_regret} \
                    model.generator.guidance_decay=${guidance_decay} \
                    model.generator.guidance_scale=${guidance_scale} \
                    model.generator.guidance_scale_pow=${guidance_scale_pow} \
                    model.generator.num_steps=${num_sample_steps}

                # Run evaluation
                python3 guided-diffusion/evaluations/evaluator.py VIRTUAL_imagenet256_labeled.npz ${output_dir}.npz

                # Clean up
                rm -rf ${output_dir}
                rm ${output_dir}.npz
                
                # Increment port for next run
                port=$((port + 1))
                
                echo "Finished run with parameters: temp=${randomize_temperature}, steps=${num_sample_steps}, scale=${guidance_scale}, noregret=${no_regret}"
                echo "---------------------------------------------------"
            done
        done
    done
done
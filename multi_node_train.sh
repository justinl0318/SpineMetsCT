#!/bin/bash
#SBATCH -t 7-00:0:0
#SBATCH -J medsam2-single
#SBATCH --mem=100G
#SBATCH -c 10
#SBATCH --gres=gpu:1
#SBATCH -o out_single.out

export PATH=/usr/local/cuda/bin:$PATH
timestamp=$(date +"%Y%m%d-%H%M")

config=configs/sam2.1_hiera_tiny_finetune512.yaml
output_path=./exp_log/single_gpu
dataset_path=/tmp2/b10902078/MEDSAM/processed_data/SpineMetsCT_npz

# Optional: export to disable DDP-related setup
unset MASTER_ADDR
unset MASTER_PORT

# ✅ Run without distributed args
CUDA_VISIBLE_DEVICES=2 python training/train.py \
        -c $config \
        --output-path $output_path \
        --dataset-path $dataset_path \
        --use-cluster 0 \
        --num-gpus 1 \
        --num-nodes 1

echo "training done"

#!/bin/sh
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=graph_partition
#SBATCH --partition=hm
#SBATCH --output=partitions/partition_log_ogbn-papers.txt

source /home/ubuntu/miniconda3/bin/activate
conda activate envforgnn
# module load anaconda3
# module load codes/gpu/cuda/11.6

python3 partition_code/partition_default.py \
                      --dataset ogbn-papers100M \
                      --num_parts 16 \
                      --balance_train \
                      --balance_edges \
                      --output partitions/ogbn-papers/metis


echo -e "\n\n============================================================================================================================================"
echo -e "============================================================================================================================================\n\n"


python3 partition_code/print_all_entropies.py \
                      --dataset "OGB-Papers" \
                      --json_metis partitions/ogbn-papers/metis/ogbn-papers100M.json \
                      --log partitions/partition_log_ogbn-papers.txt \
                      --no_of_part 16

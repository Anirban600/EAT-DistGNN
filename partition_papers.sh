#!/bin/sh
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=graph_partition
#SBATCH --partition=hm
#SBATCH --output=partitions/partition_log_ogbn-papers.txt

#conda activate envforgnn
module load anaconda3
module load codes/gpu/cuda/11.6

python3 partition_code/partition_default.py \
                      --dataset ogbn-papers100M \
                      --num_parts 16 \
                      --balance_train \
                      --balance_edges \
                      --output partitions/ogbn-papers/metis


echo -e "\n\n============================================================================================================================================"
echo -e "============================================================================================================================================\n\n"


python3 partition_code/partition_entropy_balance.py \
                      --dataset ogbn-papers100M \
                      --num_parts 48 \
                      --balance_train \
                      --balance_edges \
                      --grp_parts 16 \
                      --num_run 15 \
                      --output partitions/ogbn-papers/entropy-balanced


echo -e "\n\n============================================================================================================================================"
echo -e "============================================================================================================================================\n\n"


python3 partition_code/print_all_entropies.py \
                      --dataset "OGB-Papers" \
                      --json_metis partitions/ogbn-papers/metis/ogbn-papers100M.json \
                      --json_eb partitions/ogbn-papers/entropy-balanced/ogbn-papers100M.json \
                      --log partitions/partition_log_ogbn-papers.txt \
                      --no_of_part 16

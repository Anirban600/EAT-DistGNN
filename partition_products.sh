#!/bin/sh
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=graph_partition
#SBATCH --partition=compute
#SBATCH --output=partitions/partition_log_ogbn-products.txt

source /home/ubuntu/miniconda3/bin/activate
conda activate envforgnn
# module load anaconda3
# module load codes/gpu/cuda/11.6

python3.9 partition_code/partition_default.py \
                      --dataset ogbn-products \
                      --num_parts 4 \
                      --balance_train \
                      --balance_edges \
                      --output partitions/ogbn-products/metis


echo -e "\n\n============================================================================================================================================"
echo -e "============================================================================================================================================\n\n"


python3.9 partition_code/partition_edge_weighted.py \
                      --dataset ogbn-products \
                      --num_parts 4 \
                      --balance_train \
                      --balance_edges \
                      --c 0.5 \
                      --output partitions/ogbn-products/edge-weighted


echo -e "\n\n============================================================================================================================================"
echo -e "============================================================================================================================================\n\n"


python3.9 partition_code/print_all_entropies.py \
                      --dataset "OGB-Products" \
                      --json_metis partitions/ogbn-products/metis/ogbn-products.json \
                      --json_ew partitions/ogbn-products/edge-weighted/ogbn-products.json \
                      --log partitions/partition_log_ogbn-products.txt \
                      --no_of_part 4

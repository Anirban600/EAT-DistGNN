#!/bin/sh
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=graph_partition
#SBATCH --partition=compute
#SBATCH --output=partitions/partition_log_yelp.txt

#conda activate envforgnn
module load anaconda3
module load codes/gpu/cuda/11.6

python3 partition_code/partition_default.py \
                      --dataset yelp \
                      --num_parts 4 \
                      --balance_train \
                      --balance_edges \
                      --output partitions/yelp/yelp_metis


echo -e "\n\n============================================================================================================================================"
echo -e "============================================================================================================================================\n\n"


python3 partition_code/partition_edge_weighted.py \
                      --dataset yelp \
                      --num_parts 4 \
                      --balance_train \
                      --balance_edges \
                      --c 0.1 \
                      --output partitions/yelp/yelp_edge_weighted


echo -e "\n\n============================================================================================================================================"
echo -e "============================================================================================================================================\n\n"


python3 partition_code/partition_entropy_balance_yelp.py \
                      --dataset yelp \
                      --num_parts 100 \
                      --balance_train \
                      --balance_edges \
                      --grp_parts 4 \
                      --num_run 15 \
                      --output partitions/yelp/yelp_entropy_balanced


echo -e "\n\n============================================================================================================================================"
echo -e "============================================================================================================================================\n\n"


python3 partition_code/print_all_yelp_entropies.py \
                      --dataset Yelp \
                      --json_metis partitions/yelp/yelp_metis/yelp.json \
                      --json_ew partitions/yelp/yelp_edge_weighted/yelp.json \
                      --json_eb partitions/yelp/yelp_entropy_balanced/yelp.json \
                      --log partitions/partition_log_yelp.txt \
                      --no_of_part 4

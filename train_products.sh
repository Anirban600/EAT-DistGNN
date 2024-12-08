#!/bin/sh
#SBATCH -N 4
#SBATCH --ntasks-per-node=2
#SBATCH --job-name=ogbn_products_exp
#SBATCH --partition=standard
#SBATCH --output=ogbn-products_logs.txt

python3.9 training_code/ip_fetch.py
sleep 5
#module load python/conda-python/3.9
#module list

#Directories to store the partitions and experiment results
mkdir -p experiments/products

#Update perimission to allow execution
chmod +x deploy_trainers.sh

#OGBN-Products METIS
./deploy_trainers.sh -G ogbn-products -P metis -n 47 -p 1.0 -d 0.5 -r 0.001 -s 15 -v default -e 100 -c 1

#OGBN-Products Edge_Weighted
./deploy_trainers.sh -G ogbn-products -S 1 -P edge-weighted -n 47 -p 0.34 -d 0.5 -r 0.004 -s 15 -v cbs+gp -e 100 -c 1

#Make Results
python3.9 make_results.py --graph_name ogbn-products
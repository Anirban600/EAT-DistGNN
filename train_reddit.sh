#!/bin/sh
#SBATCH -N 4
#SBATCH --ntasks-per-node=2
#SBATCH --job-name=reddit_exp
#SBATCH --partition=standard
#SBATCH --output=reddit_logs.txt

python3 training_code/ip_fetch.py
sleep 5
#module load python/conda-python/3.9
#module list

#Directories to store the partitions and experiment results
mkdir -p experiments/reddit

#Update perimission to allow execution
chmod +x deploy_trainers.sh

#Reddit METIS
./deploy_trainers.sh -G reddit -P metis -n 41 -p 1.0 -d 0.5 -r 0.003 -s 15 -v default -e 100 -c 1

#Reddit Edge_Weighted
./deploy_trainers.sh -G reddit -S 1 -P edge-weighted -n 41 -p 0.34 -d 0.5 -r 0.001 -s 15 -v cbs+gp -e 100 -c 1

#Reddit Entropy_Balanced
./deploy_trainers.sh -G reddit -S 1 -P entropy-balanced -n 41 -p 0.34 -d 0.5 -r 0.001 -s 15 -v cbs+gp+fl -e 100 -c 0 -g 0.1

#Make Results
python3 make_results.py --graph_name reddit
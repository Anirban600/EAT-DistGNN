#!/bin/sh
#SBATCH -N 4
#SBATCH --ntasks-per-node=2
#SBATCH --job-name=yelp_exp
#SBATCH --partition=standard
#SBATCH --output=yelp_logs.txt

python3.9 training_code/ip_fetch.py
sleep 5
#module load python/conda-python/3.9
#module list

#Directories to store the partitions and experiment results
mkdir -p experiments/yelp

#Update perimission to allow execution
chmod +x deploy_trainers.sh

#Create all partitions
./partition_yelp.sh > ./partitions/partition_log_yelp.txt

#Yelp METIS
./deploy_trainers.sh -G yelp -P metis -n 100 -p 1.0 -d 0.1 -r 0.0001 -s 20 -v default -e 100 -c 1

#Yelp Edge_Weighted
./deploy_trainers.sh -G yelp -S 1 -P edge-weighted -n 100 -p 0.34 -d 0.1 -r 0.001 -s 20 -v cbs+gp -e 300 -c 1

#Make Results
python3.9 make_results.py --graph_name yelp
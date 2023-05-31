#Directories to store the partitions and experiment results
mkdir -p ./partitions
mkdir -p ./experiments/reddit

#Update perimission to allow execution
chmod +x partition_reddit.sh
chmod +x deploy_trainers.sh

#Create all partitions
./partition_flickr.sh > ./partitions/partition_log_reddit.txt

#Reddit METIS
./deploy_trainers.sh -G reddit -P metis -n 41 -p 1.0 -d 0.5 -r 0.003 -s 15 -v default -e 100 -c 1

#Reddit Edge_Weighted
./deploy_trainers.sh -G reddit -S 1 -P edge-weighted -n 41 -p 0.34 -d 0.5 -r 0.001 -s 15 -v cbs+gp -e 100 -c 1

#Reddit Entropy_Balanced
./deploy_trainers.sh -G reddit -S 1 -P entropy-balanced -n 41 -p 0.34 -d 0.5 -r 0.001 -s 15 -v cbs+gp+fl -e 100 -c 0 -g 0.1

#Make Results
python3 make_results.py --graph_name reddit
Directories to store the partitions and experiment results
mkdir -p ./partitions
mkdir -p ./experiments/flickr

#Update perimission to allow execution
chmod +x partition_flickr.sh
chmod +x deploy_trainers.sh

#Create all partitions
./partition_flickr.sh > ./partitions/partition_log_flickr.txt

#Flickr METIS
./deploy_trainers.sh -G flickr -P metis -n 7 -p 1.0 -d 0.5 -s 15 -v default -e 100 -c 1

#Flickr Edge_Weighted
./deploy_trainers.sh -G flickr -P edge-weighted -n 7 -p 0.34 -d 0.5 -s 15 -v gp+fl -e 100 -c 1

#Flickr Entropy_Balanced
./deploy_trainers.sh -G flickr -P entropy-balanced -n 7 -p 0.34 -d 0.5 -s 15 -v gp+fl -e 100 -c 0

#Make Results
python3 make_results.py --graph_name flickr
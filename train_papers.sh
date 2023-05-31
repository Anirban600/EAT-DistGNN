#Directories to store the partitions and experiment results
mkdir -p ./partitions
mkdir -p ./experiments/papers

#Update perimission to allow execution
chmod +x partition_papers.sh
chmod +x deploy_trainers.sh

#Create all partitions
./partition_flickr.sh > ./partitions/partition_log_papers.txt

#Reddit default
./deploy_trainers.sh -G papers -P metis -n 172 -p 1.0 -d 0.5 -r 0.003 -s 15 -v default -e 100 -c 1

#Reddit gp
./deploy_trainers.sh -G papers -P metis -n 172 -p 0.34 -d 0.5 -r 0.001 -s 15 -v gp -e 100 -c 1

#Reddit gp+fl
./deploy_trainers.sh -G papers -P metis -n 172 -p 0.34 -d 0.5 -r 0.001 -s 15 -v gp+fl -e 100 -c 0 -g 0.1

#Make Results
python3 make_results.py --graph_name yelp
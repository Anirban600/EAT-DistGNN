#Directories to store the partitions and experiment results
mkdir -p ./partitions
mkdir -p ./experiments/papers

#Update perimission to allow execution
chmod +x partition_papers.sh
chmod +x deploy_trainers.sh

#Create all partitions
./partition_papers.sh > ./partitions/partition_log_ogbn-papers.txt

#Reddit default
./deploy_trainers.sh -G papers -P metis -n 172 -p 1.0 -d 0.1 -r 0.01 -v default -e 60 -c 1 -l 0.01 -b 4000 -h 256

#Reddit gp
./deploy_trainers.sh -G papers -P metis -n 172 -p 0.5 -d 0.1 -r 0.01 -v gp -e 60 -c 1 -l 0.01 -b 4000 -h 256

#Reddit gp+fl
./deploy_trainers.sh -G papers -P metis -n 172 -p 0.5 -d 0.1 -r 0.01 -v gp+fl -e 60 -c 1 -g 0.2 -l 0.01 -b 4000 -h 256

#Make Results
python3 make_results.py --graph_name papers
#default
training_file="training_code/train_dist_genper_regularized.py"
ip_config_file="/root/EAT-DistGNN/training_code/ip_config1234_30054.txt"
num_epochs=100
batch_size=1000
hidden=512
num_layers=2
dropout=0.5
n_classes=40
fanout="25,25"
lr=0.0001
gamma=0
lambda=0.00001
genper_ratio=1.0
early_stop=0
sampler=0
graph_name="flickr"
partition_type="metis"
version=0
tune=0
stopping_criteria=1

#args
while getopts T:I:G:S:P:e:b:h:l:d:n:f:r:g:p:s:v:m:t:c: flag
do
    case "${flag}" in
        T) training_file=${OPTARG};;
        I) ip_config_file=${OPTARG};;
        G) graph_name=${OPTARG};;
        S) sampler=${OPTARG};;
        P) partition_type=${OPTARG};;
        e) num_epochs=${OPTARG};;
        b) batch_size=${OPTARG};;
        h) hidden=${OPTARG};;
        l) num_layers=${OPTARG};;
        d) dropout=${OPTARG};;
        n) n_classes=${OPTARG};;
        f) fanout=${OPTARG};;
        r) lr=${OPTARG};;
        g) gamma=${OPTARG};;
        p) genper_ratio=${OPTARG};;
        s) early_stop=${OPTARG};;
        v) version=${OPTARG};;
        m) lambda=${OPTARG};;
        t) tune=${OPTARG};;
        c) stopping_criteria=${OPTARG};;
    esac
done
gp=0
if (( $(echo "$genper_ratio < 1" |bc -l) )); then
    gp=1
fi

partition_path="partitions/$graph_name/$partition_type/$graph_name.json"
exp_name="${partition_type}_${version}"
mkdir -p "experiments/$graph_name/$exp_name/results"
metrics_path="experiments/$graph_name/$exp_name/results"

# source /home/ubuntu/miniconda3/bin/activate; conda activate envforgnn
# source /etc/profile.d/modules.sh; module load anaconda3
python3.9 training_code/launch_training.py \
        --workspace $(pwd) \
        --num_trainers 1 \
        --num_samplers 0 \
        --num_servers 1 \
        --num_omp_threads 1 \
        --part_config ${partition_path} \
        --ip_config ${ip_config_file} \
            "python3.9 $training_file \
            --graph_name $graph_name \
            --ip_config $ip_config_file \
            --num_epochs $num_epochs \
            --batch_size $batch_size \
            --num_hidden $hidden \
            --num_layers $num_layers \
            --dropout $dropout \
            --n_classes $n_classes \
            --fan_out $fanout \
            --lr $lr \
            --gamma $gamma \
            --llambda $lambda \
            --genper_ratio $genper_ratio \
            --early_stop $early_stop \
            --sampler $sampler \
            --metrics_path $metrics_path \
            --tune $tune \
            --stopping_criteria $stopping_criteria" > "experiments/$graph_name/$exp_name/logs.txt"

hf="experiments/$graph_name/$exp_name/hyperparams.txt"
touch $hf
echo "num_layers=$num_layers" >> "$hf"
echo "num_epochs=$num_epochs" >> "$hf"
echo "batch_size=$batch_size" >> "$hf"
echo "num_hidden=$hidden" >> "$hf"
echo "dropout=$dropout" >> "$hf"
echo "n_classes=$n_classes" >> "$hf"
echo "fanout=$fanout" >> "$hf"
echo "lr=$lr" >> "$hf"
echo "gamma=$gamma" >> "$hf"
echo "lambda=$lambda" >> "$hf"
echo "genper_ratio=$genper_ratio" >> "$hf"
echo "early_stop=$early_stop" >> "$hf"
echo "stopping_criteria=$stopping_criteria" >> "$hf"
echo "Sampler=$sampler" >> "$hf"
echo "Partition=$partition_code" >> "$hf"


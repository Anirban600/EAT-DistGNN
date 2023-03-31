#default
user="dhruv"
training_file="train_dist_genper_regularized.py"
ip_config_file="ip_config1234_30054.txt"
num_epochs=100
batch_size=1000
hidden=512
num_layers=2
dropout=0.5
n_classes=41
fanout="25,25"
lr=0.003
gamma=0
genper_ratio=1.0
early_stop=0
ensemble=0
cs=0
graph_name="reddit"
partition_code=0
server_name="long_live"
version=0

#args
while getopts U:T:I:G:S:P:E:C:e:b:h:l:d:n:f:r:g:p:s:v: flag
do
    case "${flag}" in
        U) user=${OPTARG};;
        T) training_file=${OPTARG};;
        I) ip_config_file=${OPTARG};;
        G) graph_name=${OPTARG};;
        S) server_name=${OPTARG};;
        P) partition_code=${OPTARG};;
        E) ensemble=${OPTARG};;
        C) cs=${OPTARG};;
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
    esac
done
gp=0
if (( $(echo "$genper_ratio < 1" |bc -l) )); then
    gp=1
fi
partition_path="graphs/$graph_name/PC$partition_code/$graph_name.json"
exp_name="$graph_name-$partition_code-0$gp$early_stop$ensemble$cs-$version"
cd "/home/ubuntu/workspace/Final_Code/$user"
mkdir $exp_name
mkdir "$exp_name/results"
mkdir "$exp_name/logs"
metrics_path="/home/ubuntu/workspace/Final_Code/$user/$exp_name/results"
logs_path="/home/ubuntu/workspace/Final_Code/$user/$exp_name/logs"
# ; module load codes/gpu/cuda/11.6
# &> ${log_path}/logs.txt
#--keep_alive --server_name ${server_name} 

source /etc/profile.d/modules.sh; module load anaconda3
python3 /home/ubuntu/workspace/Final_Code/launch_training.py \
        --workspace /home/ubuntu/workspace/Final_Code/ \
        --num_trainers 1 \
        --num_samplers 0 \
        --num_servers 1 \
        --num_omp_threads 1 \
        --part_config ${partition_path} \
        --ip_config ${ip_config_file} \
            "python $training_file \
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
            --genper_ratio $genper_ratio \
            --early_stop $early_stop \
            --ensemble $ensemble \
            --c_and_s $cs \
            --metrics_path $metrics_path"

hf="$exp_name/hyperparams.txt"
touch $hf
echo "num_layers=$num_layers" >> "$hf"
echo "num_epochs=$num_epochs" >> "$hf"
echo "batch_size=$batch_size" >> "$hf"
echo "num_hidden=$num_hidden" >> "$hf"
echo "dropout=$dropout" >> "$hf"
echo "n_classes=$n_classes" >> "$hf"
echo "fanout=$fanout" >> "$hf"
echo "lr=$lr" >> "$hf"
echo "gamma=$gamma" >> "$hf"
echo "genper_ratio=$genper_ratio" >> "$hf"

cd -
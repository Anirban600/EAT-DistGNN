# EAT-DistGNN

## Table of Contents
- [Overview](#overview)
- [Dataset Used](#dataset-used)
- [Setup Environment](#setup-environment)
- [Make Partitions](#make-partitions)
- [Training](#training)

## Overview
<p style="text-align: justify;">
Graph Neural Networks (GNNs) are powerful models for learning over graphs. To speed up training on very large, real-world graphs (billion scale edges) several distributed frameworks have been developed. A fundamental step in every distributed GNN framework is graph partitioning. On several benchmarks, we observe that these partitions have heterogeneous data distributions which affect model convergence and performance. They also suffer from class imbalance and out-of-distribution problems, resulting in lower performance than centralized implementations.
</p>
We holistically address these challenges, by developing entropy-aware partitioning algorithms that minimize total entropy and/or balance the entropies of graph partitions. We observe that by minimizing the average entropy of the partitions, the micro average F1 score (accuracy) can be improved. Similarly, by minimizing the variance of the entropies of the partitions and implementing a class-balanced sampler with Focal Loss, the macro average F1 score can be improved. We divide the training into a synchronous, model generalization phase, followed by an asynchronous, personalization phase that adapts each compute host's models to their local data distributions. This boosts all performance metrics and also speeds up the training process significantly.

We have implemented our algorithms on the DistDGL framework where we achieved a 4% improvement on average in micro-F1 scores and 11.6% improvement on average in the macro-F1 scores of 5 large graph benchmarks compared to the standard baselines. 


## Dataset Used

We have used five datasets from different domains. The details about them are as follows

1. **Flickr:** Flickr dataset is an image-based graph dataset where images are represented by nodes and the edges between nodes are constructed by examining the similarity of image properties (like tag, location etc).
2. **Yelp:** Yelp dataset is a community based dataset where each node is representing an human entity and their features are based on emedding-space representation of their reviews. The edges between nodes is determined by the fact whether the users are friend.
3. **Reddit:** Reddit dataset is also a community based dataset where nodes are representing posts in a "subreddit". They are labelled based on 50 simplified "subreddits" or communities. There will be an edge between two nodes or posts if an user commented on both of them.
4. **OGB-Products:** OGB-Products is an Amazon product co-purchasing network where nodes are representing products on amazon and there will be an edge if the products are purchased together.
5. **OGB-Papers:** OGB-Papers is a directed citation graph based on 111 million research papers. Here nodes are papers and directed edges are citation of one paper by another.



| **Data Set**      | **Nodes** | **Edges** | **Features** | **Labels** | **Train/Val/Test \%** | **Avg. Degree** | **Comments**   |
|:--------------------------:|:------------------:|:------------------:|:---------------------:|:-------------------:|:------------------------------:|:------------------------:|:-----------------------:|
| **Flickr**        | 89,250             | 899,756            | 500                   | 7                   | 50/25/25                       | 20                       | Noisy Labels            |
| **Yelp**          | 716,847            | 13,954,819         | 300                   | 100                 | 75/15/10                       | 10                       | Multilabel              |
| **Reddit**        | 232,965            | 114,615,892        | 602                   | 41                  | 66/10/24                       | 50                       | High Feature Dimensions |
| **OGBN-Products** | 2,449,029          | 61,859,140         | 100                   | 47                  | 8/2/90                         | 51                       | Out of Distribution     |
| **OGBN-Papers**   | 111,059,956        | 1,615,685,872      | 128                   | 172                 | 1.087/0.113/0.193              | 29                       | ~98% Unlabelled         |


## Setup Environment


## Make Partitions

Before training, we first need to break the dataset into multiple partitions using the following procedure.

### Run the Script Files:

We have dedicated script files for each dataset in the root directory. The files are

- `partition_flickr.sh`
- `partition_papers.sh`
- `partition_products.sh`
- `partition_reddit.sh`
- `partition_yelp.sh`

Each sript submits a job using SLURM commands. The jobs require one standard compute-host (high-memory host for ogbn-papers dataset) in the system. Next it activates the enviroment as described in the [Setup Environment](#setup-emvironment) section. Please update the following configurations according to your system.

- Compute host type (eg. standard, compute, hm etc.)
- Script to activate environment (if required).

Now, we can submit the job to make partitions for dataset `<dataset>` using the command
```
sbatch partition_<dataset>.sh
```

like, to make partition of `flickr`, we can run
```
sbatch partition_flickr.sh
```

While running partition script of the desired dataset say `partition_<dataset>.sh`, it creates a folder with name `<dataset>` in the `partitions` directory.


### Log files of partitions:

While running the scripts, it parallelly create log files for each dataset seperately into the `partitions` directory with name as `partition_log_<dataset>.txt`. The log file contains the execution status reports of the three algorithms used in partitioning i.e. METIS, Edge-Weighted and Entropy-Balanced following order mentioned and seperated by lines.

At the end of log file, a table is generated gathering the entropies and execution times of all algorithms as reported in the Table 4 in the [paper](https://).

```
partition_log_<dataset>.txt

      
      <log report of METIS partition>
      
      --------------------------------------------------
      --------------------------------------------------
      
      <log report of Edge-Weighted partition>
      
      --------------------------------------------------
      --------------------------------------------------
      
      <log report of Entropy-Balanced partition>
      
      --------------------------------------------------
      --------------------------------------------------
      
      <Table containing entropy and execution time>
```

### Location of the partitions:

After running the partition script, it saves the partitioned graph into a folder named like `<dataset>_<algorithm>` in the path `/partitions/<dataset>/`. So the `/partitions/<dataset>` folder liiks like,

      <dataset>/
      ├── metis/
      ├── edge-weighted/
      └── entropy-balanced/

### Directory Structure of `/partitions` folder

After running all scripts, the directory structure of `/partitions` will look like, (for illustration, we only showed directory tree of `flickr` dataset, structure of other folders will be similar).

      partitions/
      ├── flickr/
      |   ├── metis
      |   |   ├── flickr.json
      |   |   ├── part0/
      |   |   ├── part1/
      |   |   ├── part2/
      |   |   └── part3/
      |   ├── edge-weighted
      |   |   ├── flickr.json
      |   |   ├── part0/
      |   |   ├── part1/
      |   |   ├── part2/
      |   |   └── part3/
      |   └── entropy-balanced
      |       ├── flickr.json
      |       ├── part0/
      |       ├── part1/
      |       ├── part2/
      |       └── part3/
      ├── reddit/
      |   └── ...
      ├── yelp/
      |   └── ...
      ├── ogbn-papers/
      |   └── ...
      ├── ogbn-products/
      |   └── ...
      ├── partition_log_flickr.txt
      ├── partition_log_reddit.txt
      ├── partition_log_yelp.txt
      ├── partition_log_ogbn-papers.txt
      └── partition_log_ogbn-products.txt

## Training

There is one script file for every dataset which runs the required partitioning and training on that dataset and produces the results in the same format as quoted in the paper. The script files by the name:
`train_<dataset_name>.sh` and can be run by giving the command `./train_<dataset_name>.sh`. **Please run this command when current working directory is EAT-DistGNN**

The script first creates the necessary partitions and then runs training on them with the best hyperparameters we obtained. Finally it makes the tables and graphs as presented in the paper. 

The following files containing the results will be generated apart from the partitioning logs. These can be found in the experiments folder after the experiments have completed running:
1. `<graph_name>_results_table.csv`

![image](https://github.com/Anirban600/EAT-DistGNN/assets/55611035/0f1d1bce-1390-4156-84e1-982e9232218f)

The first 3 rows are the results for one graph as reported in Table 2: Comparing performance metrics of various algorithms for different graph datasets. Scores are reported as percentages and the last row gives results for that graph as reported in Table 6: Wall clock times (in sec) across all the 4 partitions of each partitioning scheme for various graph datasets.

2. `metrics_compared_<graph_name>.jpg

![image](https://github.com/Anirban600/EAT-DistGNN/assets/55611035/1cd00b82-04c0-4fb5-b131-5e32f46bf917)

This is one column of graph as presented in Figure 6: The convergence curves for training loss, validation micro and macro-F1 scores for Flickr, Reddit, Yelp and OGBNProducts using varoius partitioning schemes.

3. `speed_plot_<graph_name>.jpg`

![image](https://github.com/Anirban600/EAT-DistGNN/assets/55611035/25d3f8da-6c9e-4f9b-b3ad-5655f7610e11)

This presents the speed plot for the particular graph dataset as presented in Figure 5: Average training speeds across all partitions of default partitioning scheme for various graph datasets.



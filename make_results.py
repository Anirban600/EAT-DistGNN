import json
import os
import math 
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def read_results(parent_folder):
    results = {}
    for f in os.listdir(parent_folder):
        if('.txt' in f):
            print(parent_folder)
            file = open(os.path.join(parent_folder,f),"r+")
            vals = file.readlines()
            vals = [float(x.strip('\n')) for x in vals]
            results[f[:-4]] = vals
    return results

def compare_results(results, experiments, graph, step):
    plt.figure(figsize=(4,10))

    linestyle = ['dotted', 'dashdot']
    color = ['k', 'b']
    graphs = ['avg_train_loss', 'avg_val_micf1']
    labels = ['Train Loss', 'Val Micro F1']
    for g in range(len(graphs)):
        plt.subplot(2, 1, g+1)
        for i in range(0,len(results),1):
            plt.plot(range(len(results[i][graphs[g]])), results[i][graphs[g]], label=experiments[i][1], linestyle=linestyle[i], color=color[i], linewidth=2)

        for l in step:
            plt.axvline(l , linestyle='dashed')

        plt.xlabel('epochs', fontsize=18)
        plt.ylabel(labels[g], fontsize=18)
        plt.legend(loc=0, fontsize = 14)
        plt.xticks(fontsize = 18)
        plt.yticks(fontsize = 18)

    plt.subplots_adjust(wspace=0.3,hspace=0.3)
    plt.savefig(f'../metrics_compared_{graph}.jpg', format='jpg', bbox_inches='tight',pad_inches = 0.2, dpi = 200)

def plot_speed(results, experiments, graph, step):
    plt.figure()
    epochs = range(len(results[1]['avg_train_speed']))
    plt.plot(range(len(results[1]['avg_train_speed'])), results[1]['avg_train_speed'], label=experiments[1][1])
    plt.axvline(step, color = 'k', linestyle='dashed')

    plt.xlabel('epochs', fontsize=12)
    plt.ylabel('Speed(samples/sec)', fontsize=12)
    plt.title('Train speed curve', fontsize=14)
    plt.legend(loc=2, fontsize=12)
    

    plt.subplots_adjust(wspace=0.3,hspace=0.3)
    plt.savefig(f'../speed_plots_{graph}.jpg', format='jpg', bbox_inches='tight',pad_inches = 0.2, dpi = 200)

def plot_results(results, parent_folder):
    epochs = range(min(len(results['train_loss_0']),len(results['train_loss_1']),len(results['train_loss_2']),len(results['train_loss_3'])))
    print(epochs)

    plt.figure(figsize=(15,8))
    # plt.rc('axes', color_cycle=['r', 'g', 'b', 'y'])

    plt.subplot(2, 2, 1)
    i = 0
    results['avg_train_micf1'] = [0]*len(epochs)
    while 'train_micf1_'+str(i) in results:
        for j in range(len(epochs)):
            results['avg_train_micf1'][j] += results['train_micf1_'+str(i)][j]
        plt.plot( range(len(results['train_micf1_'+str(i)])), results['train_micf1_'+str(i)], label='Train micro rank-'+str(i))
        i+=1
    for j in range(len(results['avg_train_micf1'])):
            results['avg_train_micf1'][j] /= i
    plt.plot(epochs, results['avg_train_micf1'], label='Avg micf1')
    plt.xlabel('epochs')
    plt.title('Training micf1')
    plt.legend(loc=0)

    plt.subplot(2, 2, 2)
    i = 0
    results['avg_train_loss'] = [0]*len(epochs)
    while 'train_loss_'+str(i) in results:
        for j in range(len(epochs)):
            results['avg_train_loss'][j] += results['train_loss_'+str(i)][j]
        plt.plot(range(len(results['train_loss_'+str(i)])), results['train_loss_'+str(i)], label='Train loss rank-'+str(i))
        i+=1
    # print(results['avg_train_loss'])
    for j in range(len(results['avg_train_loss'])):
        results['avg_train_loss'][j] /= i
    # print(results['avg_train_loss'])
    plt.plot(epochs, results['avg_train_loss'], label='Avg Train loss')
    plt.xlabel('epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend(loc=0)

    plt.subplot(2, 2, 3)
    i = 0
    results['avg_val_micf1'] = [0]*len(epochs)
    while 'val_micf1_'+str(i) in results:
        for j in range(len(epochs)):
            results['avg_val_micf1'][j] += results['val_micf1_'+str(i)][j]
        plt.plot(range(len( results['val_micf1_'+str(i)])), results['val_micf1_'+str(i)], label='Val micf1 rank-'+ str(i))
        i+=1
    for j in range(len(results['avg_val_micf1'])):
            results['avg_val_micf1'][j] /= i
    plt.plot(epochs, results['avg_val_micf1'], label='Avg Val micf1')
    # plt.ylim(35, 100)
    plt.xlabel('epochs')
    plt.ylabel('micf1')
    plt.title('Val micf1')
    plt.legend(loc=0)

    plt.subplot(2, 2, 4)
    i = 0
    results['avg_train_speed'] = [0]*len(epochs)
    while 'train_speed_'+str(i) in results:
        for j in range(len(epochs)):
            results['avg_train_speed'][j] += results['train_speed_'+str(i)][j]
        plt.plot(range(len( results['train_speed_'+str(i)])), results['train_speed_'+str(i)], label='Train Speed rank-'+ str(i))
        i+=1
    for j in range(len(results['avg_train_speed'])):
            results['avg_train_speed'][j] /= i
    plt.plot(epochs, results['avg_train_speed'], label='Avg Train Speed')
    # plt.ylim(35, 100)
    plt.xlabel('epochs')
    plt.ylabel('Speed (samples/sec)')
    plt.title('Speed')
    plt.legend(loc=0)

    plt.subplots_adjust(wspace=0.5,hspace=0.5)
    plt.savefig(os.path.join(parent_folder[0],f'metrics_combined.jpg'))
    return results

def get_step(results):
    step = []
    for exp in results:
        prev_avg=exp['avg_train_speed'][0]
        for i in range(1, len(exp['avg_train_speed'])):
            if(exp['avg_train_speed'][i] - prev_avg > 50):
                step.append(i)
                break
    return step
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="results_table")
    parser.add_argument("--graph_name", type=str, help="graph name")
    args = parser.parse_args()
    os.chdir(f'./experiments/{args.graph_name}')
    # print(os.getcwd())
    # print(os.listdir())
    mapv = {
        'metis': 'DistDGL',
        'edge-weighted': 'EW+CBS+GP',
    }

    mapp = ['DistDGL', 'GP', 'GP+FL']

    mapm = {
        'best_acc': 0,
        'best_wgt': 1,
        'train_time': 2
    }

    table = {
        'Metric': ['Micro F1', 'Weighted F1', 'Train Time(s)'],
        'DistDGL': [0, 0, 0],
        'EW+CBS+GP': [0, 0, 0],
    }

    if args.graph_name == 'papers':
        i=0
        for exp in os.listdir():
            # print(f'{exp}/results/summary.json')
            with open(f'{exp}/results/summary.json', 'r',encoding='utf-8') as f:
                # print(f)
                data = json.load(f)
                for k in data.keys():
                    if k in mapm:
                        table[mapp[i]][mapm[k]] = data[k]
            i+=1

    else:     
        for exp in os.listdir():
            # print(f'{exp}/results/summary.json')
            with open(f'{exp}/results/summary.json', 'r',encoding='utf-8') as f:
                # print(f)
                data = json.load(f)
                partition = exp.split('_')[0]
                for k in data.keys():
                    if k in mapm:
                        table[mapv[partition]][mapm[k]] = data[k]

        df = pd.DataFrame(table)

        print(df)

        df.to_csv(f'../{args.graph_name}_results_table.csv')

        experiment_folders = [(f, mapv[f.split('_')[0]]) for f in os.listdir()]

        results = []
        for exp in experiment_folders:
            results.append(read_results(f'{exp[0]}/results'))

        for i in range(len(experiment_folders)):
            results[i] = plot_results(results[i], experiment_folders[i])
        step = get_step(results)

        compare_results(results, experiment_folders, args.graph_name, set(step))
        plot_speed(results, experiment_folders, args.graph_name, step[1])

import matplotlib.pyplot as plt


def save_metrics(metrics, metrics_path, rank):
    for metric in metrics.keys():
        f = open(f'{metrics_path}/{metric}_{rank}.txt', "w+")
        for i in metrics[metric]:
            f.write("%f\r\n" % (i))
        f.close()


def plot_graphs(metrics, metrics_path, rank):
    epochs = range(len(metrics['train_loss']))
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.plot(epochs, metrics['train_loss'], 'r', label='Training loss')
    plt.xlabel('epochs')
    plt.title('Training loss')
    plt.legend(loc=0)

    plt.subplot(2, 2, 2)
    plt.plot(epochs, metrics['train_speed'], 'g', label='Training speed')
    plt.xlabel('epochs')
    plt.title('Training Speed')
    plt.legend(loc=0)

    plt.subplot(2, 2, 3)
    plt.plot(epochs, metrics['train_macf1'], 'r', label='Training macro f1')
    plt.plot(epochs, metrics['train_micf1'], 'b', label='Training micro f1')
    plt.xlabel('epochs')
    plt.title('Training mac-mic f1')
    plt.legend(loc=0)

    epochs = range(0, len(metrics['val_macf1']))

    plt.subplot(2, 2, 4)
    plt.plot(epochs, metrics['val_macf1'], 'r', label='Val macro f1')
    plt.plot(epochs, metrics['val_micf1'], 'b', label='Val micro f1')
    plt.xlabel('epochs')
    plt.title('Val mac-mic f1')
    plt.legend(loc=0)
    plt.savefig(metrics_path+'/metrics_'+str(rank)+'.png')
    plt.show()

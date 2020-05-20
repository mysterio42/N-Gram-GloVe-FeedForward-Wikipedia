import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt
import os

FIGURES_DIR = 'figures/'

plt.rcParams['figure.figsize'] = (13.66, 6.79)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 100


def plot_costs(costs, nm):
    if not os.path.exists('figures'):
        os.makedirs('figures')

    plt.plot(costs)
    plt.title(f'{nm} Training Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(FIGURES_DIR + 'Figure_training' + '.png')


def plot_countries(Z, words, name):
    if not os.path.exists('figures'):
        os.makedirs('figures')

    plt.title(f'Visualize analogies with TSNE  model weight: {name}')
    plt.scatter(Z[:, 0], Z[:, 1])
    for i in range(len(words)):
        plt.annotate(s=words[i], xy=(Z[i, 0], Z[i, 1]))
    plt.savefig(FIGURES_DIR + f'Figure_visualize_{name}' + '.png')

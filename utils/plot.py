import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt
import os

FIGURES_DIR = 'figure/'

plt.rcParams['figure.figsize'] = (13.66, 6.79)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 100


def plot_costs(costs, nm):
    if not os.path.exists('figure'):
        os.makedirs('figure')

    plt.plot(costs)
    plt.title(f'{nm} Training Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(FIGURES_DIR + 'Figure_training' + '.png')
    plt.show()

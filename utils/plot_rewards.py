import matplotlib.pyplot as plt
import os
import sys
from utils import *
from matplotlib.ticker import MaxNLocator

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def plot_r(rewards, r_string, env_name):
    f = plt.figure()
    ax = f.gca()
    ax.plot(rewards)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    plt.xlabel("Iteration")
    plt.ylabel("Reward")
    plt.title("Average " + r_string + " Reward over Iterations")
    plt.show()
    f.savefig(os.path.join(assets_dir(), 'plots/{}_{}_reward.pdf'.format(env_name, r_string)))

import matplotlib.pyplot as plt
import os
import sys
from utils import *
from matplotlib.ticker import MaxNLocator

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def plot_r(rewards, r_string, env_name, algo="gail"):
    f = plt.figure()
    ax = f.gca()
    ax.plot(rewards)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    plt.xlabel("Iteration")
    plt.ylabel("Reward")
    plt.title("Average " + r_string + " Reward over Iterations")
    plt.show()
    f.savefig(os.path.join(assets_dir(), 'plots/{}_{}_{}_reward.pdf'.format(env_name, r_string, algo)))

def plot_reached(iters, rewards, env_name, extra, algo="gail"):
    f = plt.figure()
    ax = f.gca()
    ax.plot(iters, rewards)
    
    plt.xlabel("Iterations of Training")
    plt.ylabel("Reward")
    plt.title("Relative Number of Reached Goals over Iterations")
    plt.show()
    f.savefig(os.path.join(assets_dir(), 'plots/{}_{}_{}_goals.pdf'.format(env_name, extra, algo)))    

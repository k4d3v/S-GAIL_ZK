import pickle
import math
import time

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from scipy.spatial.distance import directed_hausdorff
from sklearn.cluster import DBSCAN
from hdbscan import HDBSCAN

# Some visualization stuff
sns.set()
plt.rcParams['figure.figsize'] = (12, 12)

# Utility Functions
color_lst = plt.rcParams['axes.prop_cycle'].by_key()['color']
color_lst.extend(['firebrick', 'olive', 'indigo', 'khaki', 'teal', 'saddlebrown',
                  'skyblue', 'coral', 'darkorange', 'lime', 'darkorchid', 'dimgray'])


def plot_trajs(atraj_lst):
    """
    Plots sin theta1 and sin theta2
    :param atraj_lst:
    :return:
    """
    for atraj in atraj_lst:
        #plt.scatter(atraj[:, 2], atraj[:, 3])
        #plt.plot(atraj[:, 2], atraj[:, 3])
        plt.scatter(atraj[:,0], atraj[:,1])
        plt.plot(atraj[:,0], atraj[:,1])
        #plt.scatter(atraj[:, 4], atraj[:, 5])
        plt.title("Sampled Expert Trajectories")
        plt.xlabel("x")
        plt.ylabel("y")
    plt.show()


def plot_cluster(traj_lst, cluster_lst, extra):
    """
    Plots given trajectories with a color that is specific for every trajectory's own cluster index.
    Outlier trajectories which are specified with -1 in `cluster_lst` are plotted dashed with black color
    """
    for traj, cluster in zip(traj_lst, cluster_lst):
        if cluster == -1:
            # Means it it a noisy trajectory, paint it black
            #plt.plot(traj[:, 2], traj[:, 3], c='k', linestyle='dashed')
            plt.plot(traj[:,0], traj[:,1], c='k', linestyle='dashed')

        else:
            #plt.plot(traj[:, 2], traj[:, 3], c=color_lst[cluster % len(color_lst)])
            plt.plot(traj[:,0], traj[:,1], c=color_lst[cluster % len(color_lst)])

    plt.title("Clustered Expert Trajectories, "+extra)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


def hausdorff(u, v):
    """
    Calc hausdorff dist between two points
    :param u:
    :param v:
    :return:
    """
    d = max(directed_hausdorff(u, v)[0], directed_hausdorff(v, u)[0])
    return d


def param_select(D, algo):
    """
    Do param selection and return best result (list with labels) after clustering
    :param D:
    :param algo:
    :return:
    """
    print(algo)

    best_clst_lst = []

    if algo == "hdbscan":
        mdl = HDBSCAN()
        best_clst_lst = mdl.fit_predict(D)

    # Chose best performing model
    elif algo == "dbscan":
        epsl = np.linspace(np.mean(D), np.max(D), num=30)
        nsampl = range(3, 30)
        n_unk = np.inf
        n_clst = np.inf

        for aeps in epsl:
            for ansampl in nsampl:
                mdl = DBSCAN(eps=aeps, min_samples=ansampl)
                cluster_lst = mdl.fit_predict(D)

                n_unk_new = (cluster_lst == -1).sum()
                n_clst_new = len(set(cluster_lst)) - 1
                if n_unk_new < n_unk and n_clst_new < n_clst:
                    best_clst_lst = cluster_lst
                    n_unk = n_unk_new
                    n_clst = n_clst_new

    print("Number of unassigned trajs: ", (best_clst_lst == -1).sum())
    print("Number of clusters (including unassigned): ", len(set(best_clst_lst)))
    print("Cluster list: ", best_clst_lst)

    return best_clst_lst


# 1 - Get and prepare dataset
print("1")

# Import dataset
# 39 34 45 49
traj_lst = pickle.load(open("test_trajs.p", "rb"))
traj_xy_lst = pickle.load(open("test_trajs_xy.p", "rb"))
print("Traj len: ", traj_lst[0].shape[0])

# Plotting
plot_trajs(traj_xy_lst)

# 2 - Trajectory segmentation
print("2")

t0 = time.time()
degree_threshold = 5

for traj_index, traj in enumerate(traj_lst):

    hold_index_lst = []
    previous_azimuth = 1000

    for point_index, point in enumerate(traj[:-1]):
        next_point = traj[point_index + 1]
        diff_vector = next_point - point
        azimuth = (math.degrees(math.atan2(*diff_vector[4:6])) + 360) % 360

        if abs(azimuth - previous_azimuth) > degree_threshold:
            hold_index_lst.append(point_index)
            previous_azimuth = azimuth
    hold_index_lst.append(traj.shape[0] - 1)  # Last point of trajectory is always added

    traj_lst[traj_index] = traj[hold_index_lst, :]
    traj_xy_lst[traj_index] = traj_xy_lst[traj_index][hold_index_lst, :]

print("Time for segmentation: ", time.time() - t0)
avg_len = np.mean([at.shape[0] for at in traj_lst])
print("Traj len: ", avg_len)

# Plotting
plot_trajs(traj_xy_lst)

# 3 - Distance matrix
print("3")

t0 = time.time()
traj_count = len(traj_lst)
print("Num of trajs: ", traj_count)
D = np.zeros((traj_count, traj_count))

# This may take a while
for i in range(traj_count):
    print("Traj " + str(i))
    for j in range(i + 1, traj_count):
        #distance = hausdorff(
        #    np.concatenate((traj_lst[i][:, :6], traj_lst[i][:, 9:]), axis=1),
        #    np.concatenate((traj_lst[j][:, :6], traj_lst[j][:, 9:]), axis=1))
        distance = hausdorff(traj_lst[i][:, :6], traj_lst[j][:, :6])
        #distance = hausdorff(traj_lst[i], traj_lst[j])
        D[i, j] = distance
        D[j, i] = distance

print("Time for distance matrix: ", time.time() - t0)

# 4 - Different clustering methods
print("4")

# hdbscan
t0 = time.time()
best_clst_lst = param_select(D, "hdbscan")
print("Time for HDBSCAN: ", time.time() - t0)
plot_cluster(traj_xy_lst, best_clst_lst, "hdbscan")

# dbscan
t0 = time.time()
best_clst_lst = param_select(D, "dbscan")
print("Time for DBSCAN: ", time.time() - t0)
plot_cluster(traj_xy_lst, best_clst_lst, "dbscan")

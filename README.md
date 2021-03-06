# PyTorch implementation of reinforcement learning algorithms, GAIL, S-GAIL
This repository contains:
1. policy gradient methods (TRPO, PPO, A2C)
2. [Generative Adversarial Imitation Learning (GAIL)](https://arxiv.org/pdf/1606.03476.pdf)
1, 2 taken from https://github.com/Khrylx/PyTorch-RL
3. [Situated Generative Adversarial Imitation Learning (S-GAIL)] based on 2.
4. Automatic clustering of trajectories, see https://github.com/Shathra/comparing-trajectory-clustering-methods

## Important notes
- To run mujoco environments, first install [mujoco-py](https://github.com/openai/mujoco-py) and [gym](https://github.com/openai/gym).
- If you have a GPU, I recommend setting the OMP_NUM_THREADS to 1 (PyTorch will create additional threads when performing computations which can damage the performance of multiprocessing. This problem is most serious with Linux, where multiprocessing can be even slower than a single thread):
- Start pycharm from console: env BAMF_DESKTOP_FILE_HINT=/var/lib/snapd/desktop/applications/pycharm-community_pycharm-community.desktop /snap/bin/pycharm-community %f
```
export OMP_NUM_THREADS=1
```
- PyBullet installation: https://github.com/benelot/pybullet-gym

## Features
* Support discrete and continous action space.
* Support multiprocessing for agent to collect samples in multiple environments simultaneously. (x8 faster than single thread)
* Fast Fisher vector product calculation. For this part, Ankur kindly wrote a [blog](http://www.telesens.co/2018/06/09/efficiently-computing-the-fisher-vector-product-in-trpo/) explaining the implementation details.
## Policy gradient methods
* [Trust Region Policy Optimization (TRPO)](https://arxiv.org/pdf/1502.05477.pdf) -> [examples/trpo_gym.py](https://github.com/Khrylx/PyTorch-RL/blob/master/examples/trpo_gym.py)
* [Proximal Policy Optimization (PPO)](https://arxiv.org/pdf/1707.06347.pdf) -> [examples/ppo_gym.py](https://github.com/Khrylx/PyTorch-RL/blob/master/examples/ppo_gym.py)
* [Synchronous A3C (A2C)](https://arxiv.org/pdf/1602.01783.pdf) -> [examples/a2c_gym.py](https://github.com/Khrylx/PyTorch-RL/blob/master/examples/a2c_gym.py)

### Example
* python examples/ppo_gym.py --env-name Hopper-v2

### Reference
* [ikostrikov/pytorch-trpo](https://github.com/ikostrikov/pytorch-trpo)
* [openai/baselines](https://github.com/openai/baselines)


## (Situated) Generative Adversarial Imitation Learning (GAIL, S-GAIL)
### To save trajectory
* python gail/save_expert_traj.py --model-path assets/learned_models/Hopper-v2_ppo.p
### To do imitation learning
* python gail/gail_gym.py --env-name Hopper-v2 --expert-traj-path assets/expert_traj/Hopper-v2_expert_traj.p
* python gail/sgail_gym.py --env-name Hopper-v2 --expert-traj-path assets/expert_traj/Hopper-v2_expert_traj.p

### Alternative (preffered) execution:
In VS Code, edit arg_parser.py for hyperparams and env, then run sgail_gym.py (You can choose a custom reward in line 195). For saving expert trajectories, the arguments can be edited in lines 16-28 of gail/save_expert_traj.py

### Other scripts
* run visualize.py will execute a learnt policy and show the environment to the user.
* run plot_goals.py will execute learnt policies and plot the number of goals reached.

## Trajectory Clustering
### To perform clustering
* run cluster_trajs.py

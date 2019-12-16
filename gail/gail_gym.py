from datetime import datetime

import gym
import pybulletgym
import os
import sys
import pickle
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import *
from arg_parser import prep_parser
from models.mlp_policy import Policy
from models.mlp_critic import Value
from models.mlp_policy_disc import DiscretePolicy
from models.mlp_discriminator import Discriminator
from torch import nn
from core.ppo import ppo_step
from core.common import estimate_advantages
from core.agent import Agent


def gail_reward(state, action, beta):
    """
    Reward based on Discriminator net output as described in GAIL
    """
    state_action = tensor(np.hstack([state, action]), dtype=dtype)
    with torch.no_grad():
        return math.log(discrim_net(state_action)[0].item())


def sgail_reward(state, action, beta):
    """
    Reward based on Discriminator and Generator net output as described in SGAIL
    """
    state_action = tensor(np.hstack([state, action]), dtype=dtype)

    aa = discrim_net(state_action)[0].item()
    a = math.log(discrim_net(state_action)[0].item())
    b = math.log(1 - discrim_net(state_action)[0].item())
    d = policy_net(state)
    c = beta * policy_net(state)[0].item()

    with torch.no_grad():
        return math.log(discrim_net(state_action)[0].item()) \
               - math.log(1 - discrim_net(state_action)[0].item()) \
               + beta * math.log(policy_net(state_action[0].item()))
        # TODO: Compute log(1-D) and beta*log(pi) (Discriminator should have a different output etc.)


def update_params(batch):
    states = torch.from_numpy(np.stack(batch.state)).to(dtype).to(device)
    actions = torch.from_numpy(np.stack(batch.action)).to(dtype).to(device)
    rewards = torch.from_numpy(np.stack(batch.reward)).to(dtype).to(device)
    masks = torch.from_numpy(np.stack(batch.mask)).to(dtype).to(device)
    with torch.no_grad():
        values = value_net(states)
        fixed_log_probs = policy_net.get_log_prob(states, actions)

    """get advantage estimation from the trajectories"""
    advantages, returns = estimate_advantages(rewards, masks, values, args.gamma, args.tau, device)

    """update discriminator using optimizer"""
    for _ in range(1):
        expert_state_actions = torch.from_numpy(expert_traj).to(dtype).to(device)
        g_o = discrim_net(torch.cat([states, actions], 1))
        e_o = discrim_net(expert_state_actions)
        optimizer_discrim.zero_grad()
        discrim_loss = discrim_criterion(g_o, ones((states.shape[0], 1), device=device)) + \
                       discrim_criterion(e_o, zeros((expert_traj.shape[0], 1), device=device))
        discrim_loss.backward()
        optimizer_discrim.step()

    """perform mini-batch PPO update on G and V"""
    optim_iter_num = int(math.ceil(states.shape[0] / optim_batch_size))
    for _ in range(optim_epochs):
        perm = np.arange(states.shape[0])
        np.random.shuffle(perm)
        perm = LongTensor(perm).to(device)

        states, actions, returns, advantages, fixed_log_probs = \
            states[perm].clone(), actions[perm].clone(), returns[perm].clone(), advantages[perm].clone(), fixed_log_probs[perm].clone()

        for i in range(optim_iter_num):
            ind = slice(i * optim_batch_size, min((i + 1) * optim_batch_size, states.shape[0]))
            states_b, actions_b, advantages_b, returns_b, fixed_log_probs_b = \
                states[ind], actions[ind], advantages[ind], returns[ind], fixed_log_probs[ind]

            ppo_step(policy_net, value_net, optimizer_policy, optimizer_value, 1, states_b, actions_b, returns_b,
                     advantages_b, fixed_log_probs_b, args.clip_epsilon, args.l2_reg)


def main_loop():
    beta = args.beta
    delta_beta = -args.w
    for i_iter in range(args.max_iter_num):
        """generate multiple trajectories that reach the minimum batch_size"""
        discrim_net.to(torch.device('cpu'))
        batch, log = agent.collect_samples(args.min_batch_size, state_min, state_max, action_min, action_max, beta)
        discrim_net.to(device)

        t0 = time.time()
        # Update params of V, G, D
        update_params(batch)
        # Modulate entropy correction param
        beta += delta_beta

        """Printing and saving"""
        t1 = time.time()

        if i_iter % args.log_interval == 0:
            print('{}\tT_sample {:.4f}\tT_update {:.4f}\texpert_R_avg {:.2f}\tR_avg {:.2f}'.format(
                i_iter, log['sample_time'], t1 - t0, log['avg_c_reward'], log['avg_reward']))

        if args.save_model_interval > 0 and (i_iter + 1) % args.save_model_interval == 0:
            to_device(torch.device('cpu'), policy_net, value_net, discrim_net)
            pickle.dump((policy_net, value_net, discrim_net), open(os.path.join(assets_dir(), 'learned_models/{}_gail.p'.format(args.env_name)), 'wb'))
            to_device(device, policy_net, value_net, discrim_net)

        """clean up gpu memory"""
        torch.cuda.empty_cache()


### Starting main procedures

# Prepare hyperparams
args = prep_parser()

"""Prepare torch"""
dtype = torch.float64
torch.set_default_dtype(dtype)
device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(args.gpu_index)

"""environment"""
env = gym.make(args.env_name)

"""seeding"""
np.random.seed(args.seed)
torch.manual_seed(args.seed)
env.seed(args.seed)

"""Directory of demos; state and action dim"""
demo_dir = "/home/developer/S-GAIL_ZK/Expert" # TODO: Relative path ok?
param_dir = "/home/developer/S-GAIL_ZK/params_MuJoCo" # TODO: Relative path ok?

atime = datetime.now()
new_dir_path = "/home/developer/S-GAIL_ZK/log_mujoco/%d-%d %d:%d" % (atime.month, atime.day, atime.hour, atime.minute) # TODO: Relative path ok?
os.mkdir(new_dir_path)
print("Make directory: " + new_dir_path)

"""Load expert trajs and encode labels"""
# State and action dim + filter
# TODO: Why different?
if args.env_name == "Reacher-v2":
    # Labels
    encodes_d = np.load(demo_dir + "encode_mujoco.npy")  # Class two has index 6392

    # States
    state_expert = np.load(demo_dir + "state_mujoco.npy")[:6392]
    action_expert = np.load(demo_dir + "action_mujoco.npy")[:6392]

    # Normalize & Get Min-Max
    state_expert_norm = min_max(state_expert, axis=0)
    action_expert_norm = min_max(action_expert, axis=0)

    state_dim = state_expert_norm.shape[1]
    is_disc_action = len(env.action_space.shape) == 0
    action_dim = env.action_space.shape[0]
    running_state = ZFilter((state_dim,), clip=5)

    # Combine state and actions into one array
    expert_traj = np.concatenate((state_expert_norm, action_expert_norm), axis=1)

    state_max = np.max(state_expert, axis=0)
    state_min = np.min(state_expert, axis=0)
    action_max = np.max(action_expert, axis=0)  # = [0.2112, 0.3219]
    action_min = np.min(action_expert, axis=0)  # = [-0.1343, -0.0819]

# load trajectory from expert
# TODO: Is running state needed?
else:
    state_dim = env.observation_space.shape[0]
    is_disc_action = len(env.action_space.shape) == 0
    action_dim = 1 if is_disc_action else env.action_space.shape[0]
    expert_traj, running_state = pickle.load(open(args.expert_traj_path, "rb"))
    expert_traj = expert_traj[:5500]
    # First 5500  trajs are class 1
    # running_reward = ZFilter((1,), demean=False, clip=10)

    state_max, state_min, action_max, action_min = None, None, None, None

"""define actor and critic"""
# Policy = Generator
if is_disc_action:  # For gridworld
    policy_net = DiscretePolicy(state_dim, env.action_space.n)
else:  # For pyBullet
    policy_net = Policy(state_dim, env.action_space.shape[0], log_std=args.log_std)

# State value fun
value_net = Value(state_dim)

# Discriminator
discrim_net = Discriminator(state_dim + action_dim)
discrim_criterion = nn.BCELoss()
to_device(device, policy_net, value_net, discrim_net, discrim_criterion)

# Define optimizers
optimizer_policy = torch.optim.Adam(policy_net.parameters(), lr=args.learning_rate)
optimizer_value = torch.optim.Adam(value_net.parameters(), lr=args.learning_rate)
optimizer_discrim = torch.optim.Adam(discrim_net.parameters(), lr=args.learning_rate)

# optimization epoch number and batch size for PPO
optim_epochs = 10  # 10
optim_batch_size = 64  # 64

"""create agent"""
agent = Agent(env, policy_net, device, custom_reward=gail_reward,
              running_state=running_state, render=args.render, num_threads=args.num_threads)

# Finally do the learning
main_loop()

import gym
import pybulletgym
import os
import sys
import pickle
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import *
from utils.get_reacher_vars import get_exp 
from utils.plot_rewards import *
from arg_parser import prep_parser
from models.mlp_policy import Policy
from models.mlp_critic import Value
from models.mlp_policy_disc import DiscretePolicy
from models.mlp_discriminator import Discriminator
from torch import nn
from core.ppo import ppo_step
from core.common import estimate_advantages
from core.agent import Agent


def gail_reward(state, action, encode=[], policy=[], beta=None):
    """
    Reward based on Discriminator net output as described in GAIL
    """
    saep = tensor(np.hstack((state, action, encode, policy)), dtype=dtype)
    with torch.no_grad():
        return -math.log(discrim_net(saep)[0].item())


def sgail_reward(state, action, encode=[], policy=[], beta=None):
    """
    Reward based on Discriminator and Generator net output as described in SGAIL
    """
    # TODO: Fix
    saep = tensor(np.hstack((state, action, encode, policy)), dtype=dtype)
    #print(math.log(policy[0]+ 1e-10))
    with torch.no_grad():
        D = discrim_net(saep)[0].item()
        return -( math.log(D) - math.log(1 - D) 
            + beta*math.log(policy[0]+ 1e-10)
            )
        # log(D) - log(1-D) + beta*log(pi) (Sure about pol.?)
        # Entropy regularization term


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
    rew_expert, rew_system, rel_goals = [], [], []

    beta = args.beta
    delta_beta = -args.w
    for i_iter in range(args.max_iter_num):
        """generate multiple trajectories that reach the minimum batch_size"""
        discrim_net.to(torch.device('cpu'))
        batch, log = agent.collect_samples(args.min_batch_size, beta)
        discrim_net.to(device)

        t0 = time.time()
        # Update params of V, G, D
        update_params(batch)
        # Modulate entropy correction param
        beta += delta_beta

        """Printing and saving"""
        t1 = time.time()
        rew_expert.append(log['avg_c_reward'])
        rew_system.append(log['avg_reward'])
        rel_goals.append(log['reached_goals']/log['goals'])

        if i_iter % args.log_interval == 0:
            print("beta: ", beta)
            print('Num goals: {}\tReached goals: {}'.format(log['goals'], log['reached_goals']))
            print('{}\tT_sample {:.4f}\tT_update {:.4f}\texpert_R_avg {:.2f}\tR_avg {:.2f}'.format(
                i_iter, log['sample_time'], t1 - t0, log['avg_c_reward'], log['avg_reward']))

        if args.save_model_interval > 0 and (i_iter + 1) % args.save_model_interval == 0:
            to_device(torch.device('cpu'), policy_net, value_net, discrim_net)
            pickle.dump((policy_net, value_net, discrim_net, running_state), 
                open(os.path.join(assets_dir(), 'learned_models/{}_gail_{}_{}.p'.format(args.env_name, "comp" if lower_dim else "full", str(i_iter+1))), 'wb'))
            to_device(device, policy_net, value_net, discrim_net)

        """clean up gpu memory"""
        torch.cuda.empty_cache()

    return rew_expert, rew_system, rel_goals


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
lower_dim = args.lower_dim < env.observation_space.shape[0]

"""seeding"""
np.random.seed(args.seed)
torch.manual_seed(args.seed)
env.seed(args.seed)

"""Load expert trajs and encode labels+other important stuff for Reacher (state compression)"""
state_dim, action_dim, is_disc_action, expert_traj, running_state, encodes_d = get_exp(env, args)
# 1200 1100 1500 1200 (5k)
# 2400 1900 2250 3450 (10k)
# 4500 4350 5150 6000 (20k)
# 5850 5100 6700 7350 (25k)
# 11250 11400 14600 12750 (50k)
# 24450 24250 27300 24000 (100k)
expert_traj = expert_traj[:5850]
encodes_d = encodes_d[:5850]
running_state.fix = True

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
agent = Agent(env, policy_net, device, custom_reward=gail_reward, targets=True,
              running_state=running_state, render=args.render, num_threads=args.num_threads, lower_dim=lower_dim)

# Finally do the learning
re, rs, rg = main_loop()

# Plot results
plot_r(re, "Expert", args.env_name)
plot_r(rs, "Environment", args.env_name)
plot_reached(range(len(rg)), rg, args.env_name, "comp" if lower_dim else "full")
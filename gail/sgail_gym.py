import gym
import pybulletgym
import os
import sys
import pickle
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import *
from utils.get_reacher_vars import get_exp
from utils.plot_rewards import plot_r
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
    with torch.no_grad():
        return -( math.log(discrim_net(saep)[0].item()) \
               - math.log(1 - discrim_net(saep)[0].item()) 
               + beta*math.log(policy[0]+ 1e-10)
               )
               #+ beta * policy_net.get_log_prob(torch.from_numpy(np.stack([state])).to(dtype), torch.from_numpy(np.stack([action])).to(dtype))[0].item())
        
        # log(D) - log(1-D) + beta*log(pi) (Sure about pol.?)
        # Entropy regularization term

def update_params(batch):
    states = torch.from_numpy(np.stack(batch.state)).to(dtype).to(device)
    actions = torch.from_numpy(np.stack(batch.action)).to(dtype).to(device)
    rewards = torch.from_numpy(np.stack(batch.reward)).to(dtype).to(device)
    masks = torch.from_numpy(np.stack(batch.mask)).to(dtype).to(device)
    g_encodes = torch.from_numpy(np.stack(batch.encode)).to(dtype).to(device)
    policies = torch.from_numpy(np.stack(batch.policy)).to(dtype).to(device)

    g_states_encodes = torch.cat((states, g_encodes), dim=1)

    with torch.no_grad():
        values = value_net(g_states_encodes)
        fixed_log_probs = policy_net.get_log_prob(g_states_encodes, actions)

    """get advantage estimation from the trajectories"""
    advantages, returns = estimate_advantages(rewards, masks, values, args.gamma, args.tau, device)

    """update discriminator using optimizer"""
    for _ in range(1):
        expert_full = torch.from_numpy(expert_traj).to(dtype).to(device)  # s,a,e,p
        # TODO: There are less samples from G than from expert! Maybe bad for classifier
        g_o = discrim_net(torch.cat([states, actions, g_encodes, policies], 1))  # Classify states that were generated (Label: 1)
        e_o = discrim_net(expert_full)  # Classify states from expert (Label: 0)
        optimizer_discrim.zero_grad()
        # Compare D output and real labels
        discrim_loss = discrim_criterion(g_o, ones((states.shape[0], 1), device=device)) + \
                       discrim_criterion(e_o, zeros((expert_traj.shape[0], 1), device=device))
        discrim_loss.backward()
        optimizer_discrim.step()

    """perform mini-batch PPO update on G and V"""
    # TODO: Fix (Idea: Use state+encode)
    optim_iter_num = int(math.ceil(g_states_encodes.shape[0] / optim_batch_size))
    for _ in range(optim_epochs):
        perm = np.arange(g_states_encodes.shape[0])
        np.random.shuffle(perm)
        perm = LongTensor(perm).to(device)

        g_states_encodes, actions, returns, advantages, fixed_log_probs = \
            g_states_encodes[perm].clone(), actions[perm].clone(), returns[perm].clone(), advantages[perm].clone(), fixed_log_probs[perm].clone()

        for i in range(optim_iter_num):
            ind = slice(i * optim_batch_size, min((i + 1) * optim_batch_size, g_states_encodes.shape[0]))
            g_states_encodes_b, actions_b, advantages_b, returns_b, fixed_log_probs_b = \
                g_states_encodes[ind], actions[ind], advantages[ind], returns[ind], fixed_log_probs[ind]

            ppo_step(policy_net, value_net, optimizer_policy, optimizer_value, 1, g_states_encodes_b, actions_b, returns_b,
                     advantages_b, fixed_log_probs_b, args.clip_epsilon, args.l2_reg)


def main_loop():
    rew_expert, rew_system = [], []

    # Labels for sampled trajectories
    encode_labels = [tuple(range(encode_dim)) for _ in range(encodes.shape[0])]
    encode_labels = list(np.array(encode_labels).flatten())

    beta = args.beta
    delta_beta = -args.w
    for i_iter in range(args.max_iter_num):
        """generate multiple trajectories that reach the minimum batch_size"""
        discrim_net.to(torch.device('cpu'))
        batch, log = agent.collect_samples(args.min_batch_size, beta, encode_list=encode_labels)
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


        if i_iter % args.log_interval == 0:
            print("beta: ", beta)
            print('{}\tT_sample {:.4f}\tT_update {:.4f}\texpert_R_avg {:.2f}\tR_avg {:.2f}'.format(
                i_iter, log['sample_time'], t1 - t0, log['avg_c_reward'], log['avg_reward']))

        if args.save_model_interval > 0 and (i_iter + 1) % args.save_model_interval == 0:
            to_device(torch.device('cpu'), policy_net, value_net, discrim_net)
            pickle.dump((policy_net, value_net, discrim_net, running_state), open(os.path.join(assets_dir(), 'learned_models/{}_sgail_{}.p'.format(args.env_name, "comp" if lower_dim else "full")), 'wb'))
            to_device(device, policy_net, value_net, discrim_net)

        """clean up gpu memory"""
        torch.cuda.empty_cache()

    return rew_expert, rew_system

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
state_dim, action_dim, is_disc_action, expert_sa, running_state, encodes = get_exp(env, args)
running_state.fix = True
encode_dim = encodes.shape[1]

"""define actor and critic"""
edim = args.encode_dim if args.encode_dim > 1 else 0
# Policy = Generator
policy_net = Policy(state_dim+edim, env.action_space.shape[0], log_std=args.log_std)

# State value fun
value_net = Value(state_dim, hidden_size=(128, 32), encode_dim=edim)

# Discriminator
discrim_net = Discriminator(state_dim + action_dim + edim + 1)  # s,a,e,p
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
agent = Agent(env, policy_net, device, custom_reward=sgail_reward,
              running_state=running_state, render=args.render, num_threads=args.num_threads, lower_dim=lower_dim)


"""Glue together expert trajs (s,a,e,p)"""
# Get policy from expert data
# Dimension: s,a,e,p
expert_traj = np.zeros((expert_sa.shape[0], state_dim+action_dim+encode_dim+1))
for t in range(expert_sa.shape[0]):
    se = torch.from_numpy(np.hstack((expert_sa[t][:state_dim], encodes[t])).reshape(1,-1)).to(dtype).to(device)
    pol = policy_net.get_policy(se, expert_sa[t][state_dim:])
    expert_traj[t] = np.hstack((expert_sa[t],encodes[t],pol))

# Finally do the learning
re, rs = main_loop()

# Plot results
plot_r(re, "Expert", args.env_name)
plot_r(rs, "Environment", args.env_name)

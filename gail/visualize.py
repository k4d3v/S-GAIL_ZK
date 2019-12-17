import argparse
import gym
import pybulletgym
import os
import sys
import pickle
import time
import copy
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from itertools import count
from utils import *
from utils.get_reacher_vars import get_exp


parser = argparse.ArgumentParser(description='Rollout learner')
parser.add_argument('--env-name', default="ReacherPyBulletEnv-v0", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--model-path', default="/home/developer/S-GAIL_ZK/assets/learned_models/ReacherPyBulletEnv-v0_gail.p", metavar='G',
                    help='name of the model') # TODO: Relative path
parser.add_argument('--render', action='store_true', default=True,
                    help='render the environment')
parser.add_argument('--seed', type=int, default=1, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--max-expert-state-num', type=int, default=10000, metavar='N',
                    help='maximal number of main iterations (default: 50000)')
args = parser.parse_args()

dtype = torch.float64
torch.set_default_dtype(dtype)
env = gym.make(args.env_name)
env.seed(args.seed)
if args.render: env.render(mode="human")
torch.manual_seed(args.seed)
is_disc_action = len(env.action_space.shape) == 0
state_dim = env.observation_space.shape[0]

policy_net, V, D = pickle.load(open(args.model_path, "rb"))

"""Directory of demos; state and action dim"""
demo_dir = "/home/developer/S-GAIL_ZK/Expert/" # TODO: Relative path ok?
param_dir = "/home/developer/S-GAIL_ZK/params_MuJoCo/" # TODO: Relative path ok?

"""Load expert trajs and encode labels+other important stuff for Reacher (state compression)"""
state_dim, action_dim, is_disc_action, expert_traj, running_state, encodes_d, state_max, state_min, action_max, action_min = get_exp(env, args, demo_dir)


# TODO: Show traj only if class 1
def main_loop():
    num_steps = 0

    for i_episode in count():
        obs = env.reset()
        state = delete(copy.copy(obs), state_min, state_max) if state_min is not None else obs
        if running_state is not None:
            state = running_state(state)

        # Determine target and current demo class
        target_pose = env.env.robot.target.pose().xyz()[:2]
        t01 = np.linalg.norm(target_pose-[-0.15,-0.15])<0.005
        if not t01: 
            continue  # Skip following lines if cordinates alternate

        reward_episode = 0
        
        for t in range(10000):
            state_var = tensor(state).unsqueeze(0).to(dtype)
            # choose mean action
            action = policy_net(state_var)[0][0].detach().numpy()
            # choose stochastic action
            # action = policy_net.select_action(state_var)[0].cpu().numpy()
            action = int(action) if is_disc_action else action.astype(np.float64)
            obs, reward, done, _ = env.step(action)
            next_state = delete(copy.copy(obs), state_min, state_max) if state_min is not None else obs
            if running_state is not None:
                state = running_state(state)

            reward_episode += reward
            num_steps += 1

            if args.render:
                time.sleep(1 / 60.)  # For human-friendly visualization
                env.render(mode="human")
            if done or num_steps >= args.max_expert_state_num:
                break

            state = next_state

        print('Episode {}\t reward: {:.2f}'.format(i_episode, reward_episode))

        if num_steps >= args.max_expert_state_num:
            break


main_loop()